import tensorflow as tf


# tensorflow/python/keras/datasets/mnist.py is referred to.
mnist = tf.keras.datasets.mnist

# https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz is loaded.
# 60000 data for train
# 10000 data for test
# shape : x_train -> (60000,28,28)	y_train -> (60000,)
# shape : x_test -> (10000,28,28)	y_test -> (10000,)
# datatype : numpy.ndarray
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Squential() : Linear stack of layers defined in tensorflow/python/keras/engine/sequential.py
# input_shape=(28,28) in Flatten() is sent to Layer class defined in tensorflow/python/keras/engine/base_layer.py
# Dense(units,activation=None,...)  defined in tensorflow/python/keras/layers/core.py
# Dropout(rate,...)  defined in tensorflow/python/keras/layers/core.py

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# compile(optimizer,loss=None,metrics=None,...)  defined in tensorflow/python/keras/engine/training.py
# optimizer aliases defined in tensorflow/python/keras/optimizers.py :
# 		sgd = SGD
# 		rmsprop = RMSprop
# 		adagrad = Adagrad
# 		adadelta = Adadelta
# 		adam = Adam
# 		adamax = Adamax
# 		nadam = Nadam
# For the loss, refer to tensorflow/python/keras/losses.py
# ex) MeanSquaredError can be chose by loss='mean_squared_error' in the argument.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit(x=None,y=None,batch_size=None,epochs=1,...) defined in tensorflow/python/keras/engine/training.py
# x: Input data
# y: Target data
model.fit(x_train, y_train, epochs=5)

# evaluate(x=None,y=None,...) defined in tensorflow/python/keras/engine/training.py
model.evaluate(x_test, y_test)
