import numpy as np
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gymnasium as gym
import battleship_envs

from customs import buildModel2

from random import shuffle

env = gym.make('battleship3-v1')

model = buildModel2()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse') # metrics?

NUM_EPISODES = 1000
EPS = 0.1
GAMMA = 0.19
AXIS = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
MAX_EPISODE_STEPS = 150

# SEEDING
np.random.seed(42)
env.reset(seed=42)

def predict(obs): # vectorize?, TODO: add AXIS ternary
    return model.predict([np.reshape(obs[0], (1, 10, 10, 2)), np.array([obs[1]])], verbose=0)

memory = []

# TRAINING
batchInputSpaces = []
batchInputSunk = []
batchOutput = []
def train(batch_size: int):
    global memory, batchInputSpaces, batchInputSunk, batchOutput

    # Training
    for i in range(len(memory) - 2):
        obsT, actionT, rewardT, doneT = memory.pop(0)
        if doneT:
            target = rewardT
        else:
            target = rewardT + GAMMA * np.max(predict(memory[0][0]))
        target_f = predict(obsT)

        prev_out = target_f[0].copy()

        target_f[0][actionT] = target

        batchInputSpaces.append(obsT[0])
        batchInputSunk.append(obsT[1])
        batchOutput.append(target_f[0])


    if len(batchInputSpaces) > batch_size*32:
        # Shuffle the data
        pairing = list(zip(batchInputSpaces, batchInputSunk, batchOutput))
        shuffle(pairing)
        batchInputSpaces, batchInputSunk, batchOutput = zip(*pairing)

        batchInputSpaces = np.array(batchInputSpaces)
        if AXIS == -1:
            batchInputSpaces = np.transpose(batchInputSpaces, (0, 2, 3, 1))

        model.fit([batchInputSpaces, np.array(batchInputSunk)], np.array(batchOutput), epochs=2, verbose=1)
        print(bad_hits / step)

        if rewardT < -10:
            assert predict(obsT)[0][actionT] < prev_out[actionT]
        
        batchInputSpaces = []
        batchInputSunk = []
        batchOutput = []

poss_moves = list(range(100))

for i in range(NUM_EPISODES):
    episode_logs = []
    if i % 10 == 0:
        print(f"""Completed {i} episodes, 
        reward over last 100: {np.mean([log['episode_reward'] for log in episode_logs[-10:]])}, 
        average episode length: {np.mean([log['nb_steps'] for log in episode_logs[-10:]])}, 
        bad hits: {np.mean([log['bad_hits'] for log in episode_logs[-10:]])}, 
        skips: {np.mean([log['skips'] for log in episode_logs[-10:]])}""")
        # EPS /= 2
    done = False
    obs = env.reset(seed=42)[0]
    episode_reward = 0
    step = 0
    bad_hits = 0
    skipped = 0

    slots_left = poss_moves.copy()
    while not done:
        k = 1
        # while True:
            # if np.random.random() < EPS:
            #     action = np.random.choice(slots_left)
            # elif k == 1:
            #     action = np.argmax(predict(obs))
            # else:
            #     action = tf.math.top_k(predict(obs)[0], k=k)[1][-1]
            # if action in slots_left:
            #     slots_left.remove(action)
            #     break
            # elif np.random.random() < EPS:
            #     break
            # k += 1
        old_hits = obs[0].copy()
        old_sunks = obs[1].copy()
        # print(old_hits)

        if np.random.random() < EPS:
            action = np.random.choice(slots_left) # ??
        else:
            action = np.argmax(predict((old_hits, old_sunks)))

        if action in slots_left:
            slots_left.remove(action)

        # assert np.all(obs[0] == old_hits)
        # assert np.all(obs[1] == old_sunks)

        obs, reward, done, info = env.step(action)
        if reward == -100:
            bad_hits += 1
        # else:
        #     pass
        # print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

        memory.append(((old_hits, old_sunks), action, reward, done))

        episode_reward += reward
        step += 1

        if step % 32 == 0:
            train(32)
        
        # If we are banging our head just skip and start over
        if step > MAX_EPISODE_STEPS:
            skipped = 1
            print(f"Skipped after {np.sum(old_hits)} hits")
            break
    
    train(step) # flush out at the end

    episode_logs.append({'episode': i, 'nb_steps': step, 'episode_reward': episode_reward, 'bad_hits': bad_hits, 'skips': skipped})
    
