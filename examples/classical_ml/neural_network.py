from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Embedding
from keras.datasets import imdb
from keras.utils import plot_model
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#
# Get data
#
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
test_data = x_test
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
# print(x_train[450])

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(sequences=x_train, maxlen=800)
x_test = sequence.pad_sequences(sequences=x_test, maxlen=800)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')
model = Sequential()
#
# prepeare model
#
model.add(Embedding(input_dim=10000, output_dim=100, input_length=800))
model.add(Dropout(0.5))
model.add(Conv1D(filters=250, kernel_size=3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(250))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#
# Training
#
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
history_dict = history.history
history_dict.keys()
#
# evaluation
#
results = model.evaluate(x_test, y_test)
print("Accuracy on test set:", results)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
#
# Plot
#
val_loss = history.history['val_loss']
loss = history.history['loss']
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']

epochs = range(1, len(accuracy) + 1)

plt.rcParams['figure.figsize'] = [10, 5]
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss', color='red')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='green')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo', label='Training acc', color='red')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc', color='green')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
#
# PREDICTION
#
model_prediction = Sequential()
model_prediction.add(Embedding(10000, 50, input_length=800))
model_prediction.add(Dropout(0.5))
model_prediction.add(Conv1D(filters=250, kernel_size=3, padding='valid', activation='relu', strides=1))
model_prediction.add(GlobalMaxPooling1D())
model_prediction.add(Dense(250))
model_prediction.add(Activation('relu'))
model_prediction.add(Dense(1))
model_prediction.add(Activation('sigmoid'))
model_prediction.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# training
#
history = model_prediction.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))
#
# Evaluation
#
results = model.evaluate(x_test, y_test)
print("Accuracy on test set:", results)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
#
# Plot
#
plt.hist(model_prediction.predict(x_test))
#
# Prediction
#
y_pred = model_prediction.predict(x_test)
prediction_is_positive = y_pred > 0.5
label_is_negative = y_test.reshape((25000, 1)) == 0

incorrect_cases = np.where(np.logical_and(prediction_is_positive, label_is_negative))[0]
# print ("All incorrect cases: ",incorrect_cases[0:])
print("Predicted score: ", len(incorrect_cases))
