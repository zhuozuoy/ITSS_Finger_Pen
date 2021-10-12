import numpy
import json

import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.models import Sequential

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Masking,Activation,BatchNormalization,Conv1D,GlobalMaxPool1D

def simpleLSTM(class_count=10,input_shape=(150,2)):
  model = Sequential()
  model.add(Masking(mask_value=0.,input_shape=input_shape))
  model.add(LSTM(256, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(256))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(class_count, name='Dense_output'))
  model.add(Activation(tf.keras.activations.softmax))
  # opt = tf.keras.optimizers.Adam()
  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
  model.summary()
  return model


def simpleCNN(class_count=10,input_shape=(150,2)):
  input = Input(shape=(input_shape))
  conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=5, padding="valid")(input)))
  conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=128, kernel_size=3, padding="valid")(input)))
  pool = GlobalMaxPool1D()(conv)
  dropfeat = Dropout(0.2)(pool)
  fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
  output = Dense(class_count, activation="softmax")(fc)
  model = Model(inputs=input, outputs=output)
  model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  model.summary()
  return model