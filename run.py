import gym
import numpy as np
from memory import rl_memory
from model import Model

NUM_TRAIN_GAMES = 25000
NUM_TEST_GAMES = 30
NUM_TEST_VISUAL_GAMES = 10
MAX_GAME_STEPS = 500
LOSS_PENALTY = -100
RANDOM_SEED = 0
TEST_FREQUENCY = 200
NUM_RANDOM_GAMES = 50
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 256
GYM_ENVIRONMENT = 'CartPole-v1'
LEARNING_RATE = 1e-2
LEARNING_RATE_MODEL = 0.005
DISCOUNT_FACTOR = 0.8
START_EPSILON = 0.5
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.999

model = Model(LEARNING_RATE_MODEL, DISCOUNT_FACTOR)
model.load_model()
env = gym.make(GYM_ENVIRONMENT)


def main():
    for i in range(NUM_TEST_GAMES):
        state = env.reset()
        for j in range(MAX_GAME_STEPS):
            env.render()
            action = np.argmax(model.predict([state.reshape((1,4))]), axis=1)[0]
            next_state, reward, done, info = env.step(action)

            state = next_state

            if done:
                break
        state = env.reset()

if __name__ == "__main__":
    main()
