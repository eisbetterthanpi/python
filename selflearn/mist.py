
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# https://github.com/keras-team/keras-io/blob/master/guides/training_with_built_in_methods.py
# python "F:\mist.py"



"""example use mean squared error with additional de-incentivize prediction values
far from 0.5, disincentive overconfidence,reduce overfitting?"""
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor
    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor

class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)

class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs  # Pass-through layer.

class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # `aggregation`: how to aggregate the per-batch values over each epoch
        self.add_metric(keras.backend.std(inputs), name="std_of_activation", aggregation="mean")
        return inputs  # Pass-through layer.

class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)

# loss function that computes the mean squared error between the real data and the predictions:
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

import os
# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)





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


# Generate dummy NumPy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# * Class weights * Sample weights
class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0,}
# Set weight "2" for class "5", making this class 2x more important
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64) # Shuffle and slice the dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)
# validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)
# Since the dataset already takes care of batching, we don't pass a `batch_size` argument.

# Dataset should return a tuple of dicts.
train_dataset = tf.data.Dataset.from_tensor_slices(
    ({"img_input": img_data, "ts_input": ts_data},
    {"score_output": score_targets, "class_output": class_targets},))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

data = {"inputs": np.random.random((3, 3)),"targets": np.random.random((3, 10)),}

















def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = ActivityRegularizationLayer()(x) # Insert activity regularization as a layer
# x = MetricLoggingLayer()(x) # Insert std logging as a layer
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
# outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

# simpler syntax?
# inputs = keras.Input(shape=(784,), name="digits")
# x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
# x2 = layers.Dense(64, activation="relu", name="dense_2")(x1)
# outputs = layers.Dense(10, name="predictions")(x2)

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)


image_input = keras.Input(shape=(32, 32, 3), name="img_input")
timeseries_input = keras.Input(shape=(None, 10), name="ts_input")
x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)
x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)
x = layers.concatenate([x1, x2])
score_output = layers.Dense(1, name="score_output")(x)
class_output = layers.Dense(5, name="class_output")(x)


model = get_uncompiled_model()
model = keras.Model(inputs=inputs, outputs=outputs)
model = keras.Model(inputs=[inputs, targets], outputs=predictions)
# multi-input, multi-output models
model = keras.Model(inputs=[image_input, timeseries_input], outputs=[score_output, class_output])










model.add_loss(tf.reduce_sum(x1) * 0.1)
model.add_metric(keras.backend.std(x1), name="std_of_activation", aggregation="mean")




def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],)
    return model

# # equivalent default
# model.compile(
#     optimizer="rmsprop",
#     loss="sparse_categorical_crossentropy",
#     metrics=["sparse_categorical_accuracy"],)


model.compile(optimizer="adam")  # No loss argument!
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), # Optimizer
    # optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=[CategoricalTruePositives()],
    metrics=[keras.metrics.SparseCategoricalAccuracy()], # List of metrics to monitor
    )

# plot model, plot are batch shapes, rather than per-sample shapes).
# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

# different losses to different outputs
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()],) # list version
    # loss={"class_output": keras.losses.CategoricalCrossentropy()},) # dict version
    # loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],)

# If only 1 loss function, the same loss function would be applied to every output, Likewise for metrics:
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    metrics=[[keras.metrics.MeanAbsolutePercentageError(),keras.metrics.MeanAbsoluteError(),],
        [keras.metrics.CategoricalAccuracy()],],)

# Since we gave names to our output layers, we could also specify per-output losses and metrics via a dict:
# add weight to output score loss 2x
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
    loss={"score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),},
    metrics={"score_output": [keras.metrics.MeanAbsolutePercentageError(),keras.metrics.MeanAbsoluteError(),],
        "class_output": [keras.metrics.CategoricalAccuracy()],},
    loss_weights={"score_output": 2.0, "class_output": 1.0},)

model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)


















callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss", # Stop training when `val_loss` is no longer improving
        min_delta=1e-2, # "no longer improving" being defined as "no better than 1e-2 less"
        patience=2, # "no longer improving" being further defined as "for at least 2 epochs"
        verbose=1,
        )]

# ModelCheckpoint` callback: save checkpoints of your model at frequent intervals.
callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}",
        # filepath="mymodel_{epoch}", # Path where to save the model
        # save_freq=100, # saves a SavedModeland training loss every 100 batches
        save_best_only=True,  # Only save a model iff `val_loss` has improved.
        monitor="val_loss", # The saved model name will include the current epoch.
        verbose=1,)]


# We need to one-hot encode the labels to use MSE
y_train_one_hot = tf.one_hot(y_train, depth=10)




model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
model.fit(x_train, y_train, batch_size=64, epochs=1)

model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)

model.fit(train_dataset, epochs=3)
model.fit(train_dataset, epochs=1, validation_data=val_dataset)

model.fit(train_dataset, epochs=3, steps_per_epoch=100) # Only use the 100 batches per epoch (that's 64 * 100 samples)

model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
model.fit(x_train, y_train, epochs=2, batch_size=64,callbacks=callbacks, validation_split=0.2)

# Only run validation using the first 10 batches of the dataset
model.fit(train_dataset,epochs=1, validation_data=val_dataset, validation_steps=10,)


# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)
# Alternatively, fit on dicts
model.fit(
    {"img_input": img_data, "ts_input": ts_data},
    {"score_output": score_targets, "class_output": class_targets},
    batch_size=32,epochs=1,)

model.fit(data)




history = model.fit(
    x_train, y_train, batch_size=64, epochs=2,
    # We pass some validation for monitoring validation loss and metrics at the end of each epoch
    validation_data=(x_val, y_val),)

# The returned "history" object holds a record of the loss values and metric values during training:
history.history



# You can also evaluate or predict on a dataset.
print("Evaluate")
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))


results = model.evaluate(x_test, y_test, batch_size=128)
# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)













def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()
model = make_or_restore_model()



# gradually reduce the learning as training progresses. aka "learning rate decay" `learning_rate`
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

### Using callbacks to implement a dynamic learning rate schedule




# command line: tensorboard --logdir=/full_path_to_your_logs
keras.callbacks.TensorBoard(
    log_dir="/full_path_to_your_logs",
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",)  # How often to write logs (default: once per epoch)

# sequence example
# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np
# # Here, `filenames` is list of path to the images
# # and `labels` are the associated labels.
# class CIFAR10Sequence(Sequence):
#     def __init__(self, filenames, labels, batch_size):
#         self.filenames, self.labels = filenames, labels
#         self.batch_size = batch_size
#     def __len__(self):
#         return int(np.ceil(len(self.filenames) / float(self.batch_size)))
#     def __getitem__(self, idx):
#         batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return np.array([
#             resize(imread(filename), (200, 200))
#                for filename in batch_x]), np.array(batch_y)
# sequence = CIFAR10Sequence(filenames, labels, batch_size)
# model.fit(sequence, epochs=10)

# python "F:\mist.py"
