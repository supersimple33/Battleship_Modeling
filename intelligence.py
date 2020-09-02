from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

import numpy as np
from random import randint
import sys
import json

training_data = []
with open("data.json", "r") as f:
    training_data = json.load(f)
    print("Data Loaded. Rows: " + str(len(training_data)))
with open("data2e.json", "r") as f:
    training_data.extend(json.load(f))
    print("Data Loaded. Rows: " + str(len(training_data)))

# print(training_data)

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(133, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))

    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    
    model.fit(X, y, epochs=10)
    return model

trained_model = train_model(training_data)

trained_model.save('saved_model/my_model.h5')

print("Done")