import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tqdm import trange

def get_model(input_shape, output_units, rnn_units, layer_cnt):
    inputs = keras.layers.Input(input_shape)
    x = inputs
    for i in range(layer_cnt):
        x = keras.layers.GRU(rnn_units, return_sequences=True)(x)
    outputs = keras.layers.Dense(output_units, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

def timegan_init(time_series_len, features, rnn_units, rnn_layers):
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
    embedder, generator, supervisor, recovery, discriminator = timegan_tuple

    #convert to float32(because random_vector's type is float32, should be matched)
    x = x.astype(np.float32)
    get_batch = lambda : x[np.random.permutation(x.shape[0])[:batch_size]]
    get_random_vector = lambda : np.random.uniform(size=(batch_size, x.shape[1], x.shape[2]))

    #loss & optimizer
    mse = keras.losses.MeanSquaredError()
    bce = keras.losses.BinaryCrossentropy()
    opt_autoencoder = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_supervisor = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_generator = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_embedder = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_discriminator = keras.optimizers.Adam(learning_rate=learning_rate)

    #train autoencoder
    print(f"train autoencoder")
    for _ in trange(epochs):
        batch = get_batch()
        with tf.GradientTape() as tape:
            y_true = embedder(batch)
            y_true = recovery(y_true)
            loss = 10 * tf.sqrt(mse(y_true, batch))
        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt_autoencoder.apply_gradients(zip(gradients, var_list))

    #train supervisor
    print(f"train supervisor")
    for _ in trange(epochs):
        batch = get_batch()
        with tf.GradientTape() as tape:
            y_true = embedder(batch)
            y_pred = supervisor(y_true)
            loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])
        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt_supervisor.apply_gradients(apply_grads)

    #joint train
    print(f"joint train")
    for _ in trange(epochs):
        for __ in range(2):
            batch = get_batch()
            random_vector = get_random_vector()
            #train generator
            with tf.GradientTape() as tape:
                #supervised loss
                y_true = embedder(batch)
                y_pred = supervisor(y_true)
                supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

                #unsupervised loss
                y_true = tf.ones((batch_size, x.shape[1], 1))
                y_pred = generator(random_vector)
                y_pred = supervisor(y_pred)
                y_pred = discriminator(y_pred)
                unsupervised_loss = bce(y_true, y_pred)

                #unsupervised loss - E
                y_true = tf.ones((batch_size, x.shape[1], 1))
                y_pred = generator(random_vector)
                y_pred = discriminator(y_pred)
                unsupervised_loss_e = bce(y_true, y_pred)

                #moment loss
                y_true = batch
                y_pred = generator(random_vector)
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
            opt_generator.apply_gradients(zip(gradients, var_list))

            #train embedder
            with tf.GradientTape() as tape:
                #supervised loss
                y_true = embedder(batch)
                y_pred = supervisor(y_true)
                supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

                #reconstruction loss
                y_true = embedder(batch)
                y_true = recovery(y_true)
                y_pred = batch
                reconstruction_loss = 10 * tf.sqrt(mse(y_true, y_pred))

                loss = reconstruction_loss + 0.1 * supervised_loss

            var_list = embedder.trainable_variables + recovery.trainable_variables
            gradients = tape.gradient(loss, var_list)
            opt_embedder.apply_gradients(zip(gradients, var_list))

        batch = get_batch()
        random_vector = get_random_vector()
        #train discriminator
        with tf.GradientTape() as tape:
            #loss on FN
            y_true = tf.ones((batch_size, x.shape[1], 1))
            y_pred = embedder(batch)
            y_pred = discriminator(y_pred)
            loss_on_FN = bce(y_true, y_pred)

            #loss on FP
            y_true = tf.zeros((batch_size, x.shape[1], 1))
            y_pred = generator(random_vector)
            y_pred = supervisor(y_pred)
            y_pred = discriminator(y_pred)
            loss_on_FP = bce(y_true, y_pred)

            #loss on FP - E
            y_true = tf.zeros((batch_size, x.shape[1], 1))
            y_pred = generator(random_vector)
            y_pred = discriminator(y_pred)
            loss_on_FP_E = bce(y_true, y_pred)           

            loss = loss_on_FN + loss_on_FP + loss_on_FP_E

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt_discriminator.apply_gradients(zip(gradients, var_list))

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
