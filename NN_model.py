import numpy as np

# Tensorflow stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

class NN:

    def __init__(self):
        # Hyper parameters
        self.lr = 0.0001
        self.alpha = 0.9

        # main model
        self.main_model = self.create_model()

    def create_model(self):
        # Create a NN
        model = keras.Sequential()
        model.add(Input(shape=(17)))
        model.add(Dense(34, activation="relu"))
        model.add(Dense(34, activation="relu"))
        model.add(Dense(4, activation="softmax"))  # softmax?
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.lr), metrics=['accuracy'])
        return model

    def set_model(self, model):
        # target model
        #self.main_model = clone_model(model)
        self.main_model.set_weights(model.get_weights())

    def get_action(self, state, move_pieces):
        # forward pass of NN (get Q values given state)
        Qs = self.main_model.predict(state, verbose=0)

        if len(move_pieces):  # If posible to move

            # Pick highest action from neural network
            out = Qs[0]
            action_list = []
            for x in move_pieces:
                action_list.append((out[x], x))
            action_list.sort(reverse=True)
            action = action_list[0][1]

        else:  # if not possible to move
            action = -1

        return action


    def train_model(self, BUFFER, reward):

        state = []
        label = []

        for s, a in BUFFER:

            Qs = self.main_model.predict(s, verbose=0)
            Q = Qs[0]

            update = Q
            update[a] = Q[a] + self.alpha * (reward - Q[a])

            state.append(s[0])
            label.append(update)

        state = np.array(state)
        label = np.array(label)

        state.reshape(17, len(state))
        label.reshape(4, len(label))

        #print(state)
        #print(label)

        print("Training")
        self.main_model.fit(state, label, batch_size=1, epochs=1, shuffle=True, verbose=1)




