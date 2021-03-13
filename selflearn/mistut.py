import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# https://www.tensorflow.org/guide/keras/train_and_evaluate/#many_built-in_optimizers_losses_and_metrics_are_available
# python "F:\mistut.py"

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)

print("Fit model on training data")
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val),
    ) # monitoring validation loss and metrics at the end of each epoch
history.history
print('huhh',history,model)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer) on new data using `predict`
predictions = model.predict(x_test[:3])
# print(model.predict(x_test[0]))
# print('tr',x_train[:3])
print('test:3',x_test[:3])
tp=tf.argmax(predictions, 1)
print(tp,tp[0])
print("predictions", predictions, predictions.shape)
model.summary()

# python "F:\mistut.py"
