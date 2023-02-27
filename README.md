# TimeGAN-keras
Implementation of Time-series Generative Adversarial Networks(TimeGAN, https://github.com/jsyoon0823/TimeGAN) with using Keras

# timegan.py
timegan's features is implemented as python(tensorflow-keras) function.

### for training
```
# x is data for training, (n, time-series-len, features) shape numpy array
timegan_tuple = timegan_init(x.shape[1], x.shape[2], rnn_units=24, rnn_layers=3)
timegan_train(x, timegan_tuple, epochs=2000, batch_size=32, learning_rate=0.0005)
syn_generator = timegan_export_generator(timegan_tuple)
```

### for generate data
```
syn_generator = timegan_export_generator(timegan_tuple)
#return (32, x.shape[1], x.shape[2]) shape generated data.
syn_data = generator_gen(syn_generator, generate_cnt=32)
```
before data-generation/model-save, must run ```timegan_export_generator(timegan_tuple)``` function to get connected-generator network.

### save/load
```
generator_save(syn_generator, save_path)
syn_generator = generator_load(save_path, time_series_len, features, rnn_units, rnn_layers)
```
above functions use ```tf.keras.model.save_weights```, so you can save/load manually.

# Goal
This project was created to conduct some experiments : optimizing TimeGAN for eye-writing data.
