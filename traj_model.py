import numpy
import json

import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.models import Sequential

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Masking

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
  model.add(Dense(class_count, activation='softmax'))
  # opt = tf.keras.optimizers.Adam()
  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
  model.summary()
  return model