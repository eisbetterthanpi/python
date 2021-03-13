# python "F:\tfktut.py"

# tensorflow_version 2.x # switch bet 1 and 2
import tensorflow as tf
# import tensorflow-gpu as tf # nope
print("#############################################")
print(tf.__version__)
# print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
# tf.config.experimental.list_physical_devices('GPU'))
print("#############################################")

model = tf.keras.Sequential()

# opt = SGD(learning_rate=0.01, momentum=0.9)
# model.compile(optimizer=opt, loss='binary_crossentropy')
#
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X, y, epochs=5, batch_size=10, verbose=0)
# loss = model.evaluate(X, y, verbose=0)
# yhat = model.predict(X)







# https://www.tensorflow.org/guide/keras/sequential_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# model = keras.Sequential([
#         layers.Dense(2, activation="relu", name="layer1"),
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(4, name="layer3"),])
# x = tf.ones((3, 3))
# y = model(x)

# model = keras.Sequential(name="my_sequential")
# model.add(layers.Dense(2, activation="relu", name="layer1"))
# model.add(layers.Dense(3, activation="relu", name="layer2"))
# model.add(layers.Dense(4, name="layer3"))
# x = tf.ones((1, 4))
# y = model(x)

# layer1 = layers.Dense(2, activation="relu", name="layer1")
# layer2 = layers.Dense(3, activation="relu", name="layer2")
# layer3 = layers.Dense(4, name="layer3")
# x = tf.ones((3, 3))
# y = layer3(layer2(layer1(x)))

# model = keras.Sequential()
# model.add(keras.Input(shape=(4,)))
# model.add(layers.Dense(2, activation="relu"))

# print(model.layers)
# model.summary()



# model = keras.Sequential()
# model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
# model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))
#
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(2))
#
# model.summary() #(None, 12, 12, 32)
# model.add(layers.GlobalMaxPooling2D()) # apply global max pooling
# model.add(layers.Dense(10)) # classification layer.



inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)




model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)


import numpy as np
data = np.load('your_file.npz')







# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# # define the model
# model = Sequential()
# model.add(Dense(100, input_shape=(8,)))
# model.add(Dense(80))
# model.add(Dense(30))



# from tensorflow.keras import Model
# from tensorflow.keras import Input
# # import tensorflow.keras
# # from tensorflow.keras import *
# from tensorflow.keras.layers import Dense
# x_in = Input(shape=(8,))
# x = Dense(10)(x_in)
# x_out = Dense(1)(x)
# model = Model(inputs=x_in, outputs=x_out)

# mlp

#######################################################################
# openai gym works
# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()


# python "F:\tfktut.py"
