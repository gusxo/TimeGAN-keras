import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tqdm import trange

def timegan_init(time_series_len, features, rnn_units, rnn_layers):
    def get_model(input_shape, output_units, rnn_units, layer_cnt):
        inputs = keras.layers.Input(input_shape)
        x = inputs
        for _ in range(layer_cnt):
            x = keras.layers.GRU(rnn_units, return_sequences=True)(x)
        outputs = keras.layers.Dense(output_units, activation="sigmoid")(x)
        return keras.Model(inputs, outputs)

    input_shape = (time_series_len, features)
    latent_code_shape = (time_series_len, rnn_units)
    embedder = get_model(input_shape, rnn_units, rnn_units, rnn_layers)
    generator = get_model(input_shape, rnn_units, rnn_units, rnn_layers)
    supervisor = get_model(latent_code_shape, rnn_units, rnn_units, rnn_layers)
    recovery = get_model(latent_code_shape, features, rnn_units, rnn_layers)
    discriminator = get_model(latent_code_shape, 1, rnn_units, rnn_layers-1)
    return embedder, generator, supervisor, recovery, discriminator

def timegan_export_generator(timegan_tuple):
    _, generator, supervisor, recovery, _ = timegan_tuple
    syn_gen_input = keras.Input(generator.input_shape[1:])
    syn_gen_output = generator(syn_gen_input)
    syn_gen_output = supervisor(syn_gen_output)
    syn_gen_output = recovery(syn_gen_output)
    return keras.Model(syn_gen_input, syn_gen_output)
    
def timegan_train(x, timegan_tuple, epochs, batch_size, learning_rate):
    #convert to float32(because random_vector's type is float32, should be matched)
    x = x.astype(np.float32)
    get_batch = lambda : tf.convert_to_tensor(x[np.random.permutation(x.shape[0])[:batch_size]])
    get_random_vector = lambda : tf.convert_to_tensor(np.random.uniform(size=(batch_size, x.shape[1], x.shape[2])))

    #loss & optimizer
    mse = keras.losses.MeanSquaredError()
    bce = keras.losses.BinaryCrossentropy()
    opt_autoencoder = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_supervisor = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_generator = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_embedder = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_discriminator = keras.optimizers.Adam(learning_rate=learning_rate)

    #training functions
    @tf.function
    def train_autoencoder(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            y_true = embedder(x)
            y_true = recovery(y_true)
            loss = 10 * tf.sqrt(mse(y_true, x))
        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def train_supervisor(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            y_true = embedder(x)
            y_pred = supervisor(y_true)
            loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])
        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return loss

    @tf.function
    def train_generator(x, z, timegan, mse, bce, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            #supervised loss
            y_true = embedder(x)
            y_pred = supervisor(y_true)
            supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

            #unsupervised loss
            y_true = tf.ones((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = supervisor(y_pred)
            y_pred = discriminator(y_pred)
            unsupervised_loss = bce(y_true, y_pred)

            #unsupervised loss - E
            y_true = tf.ones((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = discriminator(y_pred)
            unsupervised_loss_e = bce(y_true, y_pred)

            #moment loss
            y_true = x
            y_pred = generator(z)
            y_pred = supervisor(y_pred)
            y_pred = recovery(y_pred)
            y_true_mean, y_true_var = tf.nn.moments(y_true, axes=[0])
            y_pred_mean, y_pred_var = tf.nn.moments(y_pred, axes=[0])
            v1 = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
            v2 = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
            moment_loss = v1 + v2

            loss = supervised_loss + 100 * tf.sqrt(unsupervised_loss) + unsupervised_loss_e + 100 * moment_loss

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def train_embedder(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            #supervised loss
            y_true = embedder(x)
            y_pred = supervisor(y_true)
            supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

            #reconstruction loss
            y_true = embedder(x)
            y_true = recovery(y_true)
            y_pred = x
            reconstruction_loss = 10 * tf.sqrt(mse(y_true, y_pred))

            loss = reconstruction_loss + 0.1 * supervised_loss

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def train_discriminator(x, z, timegan, bce, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            #loss on FN
            y_true = tf.ones((x.shape[0], x.shape[1], 1))
            y_pred = embedder(x)
            y_pred = discriminator(y_pred)
            loss_on_FN = bce(y_true, y_pred)

            #loss on FP
            y_true = tf.zeros((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = supervisor(y_pred)
            y_pred = discriminator(y_pred)
            loss_on_FP = bce(y_true, y_pred)

            #loss on FP - E
            y_true = tf.zeros((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = discriminator(y_pred)
            loss_on_FP_E = bce(y_true, y_pred)           

            loss = loss_on_FN + loss_on_FP + loss_on_FP_E

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    
    #conduct train
    print(f"train autoencoder")
    for _ in trange(epochs):
        batch = get_batch()
        train_autoencoder(batch, timegan_tuple, mse, opt_autoencoder)

    print(f"train supervisor")
    for _ in trange(epochs):
        batch = get_batch()
        train_supervisor(batch, timegan_tuple, mse, opt_supervisor)

    print(f"joint train")
    for _ in trange(epochs):
        for __ in range(2):
            batch = get_batch()
            random_vector = get_random_vector()
            train_generator(batch, random_vector, timegan_tuple, mse, bce, opt_generator)
            train_embedder(batch, timegan_tuple, mse, opt_embedder)

        batch = get_batch()
        random_vector = get_random_vector()
        train_discriminator(batch, random_vector, timegan_tuple, bce, opt_discriminator)

    return timegan_tuple

def generator_save(syn_gen, save_path):
    syn_gen.save_weights(save_path)

def generator_load(save_path, time_series_len, features, rnn_units, rnn_layers):
    timegan = timegan_init(time_series_len, features, rnn_units, rnn_layers)
    syn_gen = timegan_export_generator(timegan)
    syn_gen.load_weights(save_path)
    return syn_gen

def generator_gen(syn_gen, generate_cnt):
    syn = syn_gen.predict(np.random.uniform(size=(generate_cnt, syn_gen.input_shape[1], syn_gen.input_shape[2])))
    return syn
