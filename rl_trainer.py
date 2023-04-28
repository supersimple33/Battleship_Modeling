import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents import DqnAgent
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.networks.q_network import QNetwork
from tf_agents.drivers import py_driver

import gymnasium as gym

from battleship2_env import Battleship2

from customs import buildModel2

NUM_EPISODES = 100
NUM_ITERS = 1000
EPS = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Seeding:
np_random = np.random.default_rng(42)

tf_env = TFPyEnvironment(Battleship2())
tf_env.reset()


model = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    # conv_layer_params=[(1, (1,1), 1), (4, (3,3), 1), (8, (5,5), 1)],
)

# model.compile(optimizer=optimizer, loss='mse') # metrics?

train_step_counter = tf.Variable(0)

agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=model,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)
avg_return = tf_metrics.AverageReturnMetric(buffer_size=10, name='AverageReturn', dtype=tf.float32)
returns = [avg_return]

# Reset the environment.
time_step = tf_env.reset()

for _ in range(NUM_EPISODES):
    # Reset the environment.
    time_step = tf_env.reset()
    avg_return.reset()
    experience = []
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        next_time_step = tf_env.step(action_step.action)
        experience.append((time_step.observation, action_step, next_time_step.observation))
        time_step = next_time_step
        # avg_return(time_step.reward, time_step.step_type)
    loss = agent.train(experience).loss
    print(loss)
    returns.append(avg_return.result())
    
