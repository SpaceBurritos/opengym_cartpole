import gym
import numpy as np
from memory import rl_memory
from model import Model

NUM_TRAIN_GAMES = 25000
NUM_TEST_GAMES = 10
NUM_TEST_VISUAL_GAMES = 10
MAX_GAME_STEPS = 500
LOSS_PENALTY = -150
RANDOM_SEED = 0
TEST_FREQUENCY = 5
NUM_RANDOM_GAMES = 5
MEMORY_CAPACITY = 300000
BATCH_SIZE = 256
GYM_ENVIRONMENT = 'CartPole-v1'
LEARNING_RATE = 1e-2
LEARNING_RATE_MODEL = 0.005
DISCOUNT_FACTOR = 0.9
START_EPSILON = 0.95
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995

model = Model(LEARNING_RATE_MODEL, DISCOUNT_FACTOR)
memory = rl_memory(MEMORY_CAPACITY)
env = gym.make(GYM_ENVIRONMENT)


def run_test(render=True):
    times = []
    for i in range(NUM_TEST_GAMES):
        state = env.reset()
        for j in range(MAX_GAME_STEPS):
            if render:
                env.render()
            action = np.argmax(model.predict([state.reshape((1,4))]), axis=1)[0]
            print(action)
            next_state, reward, done, info = env.step(action)

            state = next_state

            if done:
                break
        times.append(j)
        state = env.reset()
    return times


def get_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        prediction = model.predict(state.reshape((1,4)))
        return np.argmax(prediction, axis=1)[0]


def run_train():
    prev_avg_time = 0
    epsilon = START_EPSILON
    for i_game in range(NUM_TRAIN_GAMES):
        state = env.reset()
        for i_episode in range(MAX_GAME_STEPS):
            #env.render()
            action = get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = LOSS_PENALTY
                if i_game > NUM_RANDOM_GAMES and epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

            model.target_train()
            state = next_state

            if done:
                break
        print(i_game)
        memory.add(state, action, reward, next_state, done)
        model.model_train(memory.get_batch(BATCH_SIZE))
        if i_game % 1 == 0:
            times = run_test()
            avg_time = sum(times) / len(times)
        if avg_time > TEST_FREQUENCY and (avg_time) > prev_avg_time:

            prev_avg_time = avg_time
            model.save_model()
        print(avg_time, prev_avg_time)
    model.save_model()

run_train()
print(run_test())
env.close()

print("DONE")
