from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np

class Model:
    def __init__(self, discount_factor, learning_rate):
        self.state_size = 4
        self.action_size = 2
        self.tau = 0.125
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = self.set_model()
        self.target_model = self.set_model()

    def set_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def predict(self, states):
        return self.model.predict(states)

    def model_train(self, data):

        states, actions, rewards, next_states, dones = data
        targets = np.zeros((len(states), 2))
        for i in range(len(states)):
            state = states[i].reshape((1,4))
            next_state = next_states[i].reshape((1,4))
            target = self.target_model.predict(state)
            if dones[i]:
                target[0][actions[i]] = rewards[i]
            else:
                Q_future = max(self.target_model.predict(next_state)[0])
                targets[i][actions[i]] = rewards[i] + Q_future * self.discount_factor
        self.model.fit(states, targets)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]*self.tau + target_weights[i]*(1-self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")
        print("saved model to disk")

    def load_model(self):
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        self.model = loaded_model

