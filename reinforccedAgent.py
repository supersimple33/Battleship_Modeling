import tensorflow as tf

import numpy as np

import time

from tf_agents.networks import network, q_network
from tf_agents.environments import tf_py_environment
from tf_agents.policies import q_policy
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from battleship2_env import Battleship2

tf.compat.v1.enable_v2_behavior()

environment = Battleship2()
tf_env = tf_py_environment.TFPyEnvironment(environment)

action_spec = tf_env.action_spec()
num_actions = action_spec.maximum - action_spec.minimum + 1

# MODEL TWEAKS
FILTERS = 64
REG = None
CHANNEL_TYPE = "channels_last"

class QNetwork(network.Network):
	def __init__(self, input_tensor_spec, action_spec, num_actions=num_actions, name=None):
		super(QNetwork, self).__init__(
			input_tensor_spec=input_tensor_spec,
			state_spec=(),
			name=name)
		self._sub_layers = [
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(288),
			tf.keras.layers.Dense(416),
			tf.keras.layers.Dense(num_actions,activation='sigmoid'), #tweak later
		]

	def call(self, inputs, step_type=None, network_state=()):
		del step_type
		inputs = tf.cast(inputs, tf.float32)
		for layer in self._sub_layers:
			inputs = layer(inputs)
		return inputs, network_state

timSpec = tf_env.time_step_spec()
obsSpec = tf_env.observation_spec()
# q_net = QNetwork(input_tensor_spec=obsSpec,action_spec=action_spec)
q_net = q_network.QNetwork(input_tensor_spec=obsSpec, batch_squash=False, action_spec=action_spec, conv_layer_params=[(32, 5, 1)], fc_layer_params=None) #, preprocessing_layers=[tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))]
q_net.create_variables(obsSpec, training=True)
print(q_net.summary())
q_policy = q_policy.QPolicy(timSpec, action_spec, q_network=q_net)
global_step = tf.compat.v1.train.get_or_create_global_step()

# AGENT DEF
agent = dqn_agent.DqnAgent(
    timSpec,
    action_spec,
	n_step_update=4, # safe?
    q_network=q_net,
    optimizer=tf.keras.optimizers.Adam(0.001),
	train_step_counter=global_step) #tf.compat.v1.train.AdamOptimizer(0.001)

# BUFFER TWEAKS
batch_size = 32
MAX_LENGTH = 1000
NUM_GAMES = 15
EPOCHS = 100

data_spec = (action_spec,obsSpec)
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec,batch_size=tf_env.batch_size,max_length=MAX_LENGTH)
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
lossAvg = tf.keras.metrics.Mean()
observers = [num_episodes, env_steps, replay_buffer.add_batch]

driver = dynamic_episode_driver.DynamicEpisodeDriver(tf_env, q_policy, observers, num_episodes=NUM_GAMES) # switch from episode to step

final_time_step, policy_state = driver.run()
# print('final_time_step', final_time_step)
# print('Number of Steps: ', env_steps.result().numpy())
# print('Number of Episodes: ', num_episodes.result().numpy())

# Decorated Function
agent.train = common.function(agent.train)#move

# dataset = replay_buffer.as_dataset(sample_batch_size=1,num_steps=2,single_deterministic_pass=True) # do I want single_deterministic sample_batch_size=1,
# iterator = iter(dataset)
# for _ in range(replay_buffer.num_frames()//2):
# 	trajectories, _ = next(iterator)
# 	loss = agent.train(experience=trajectories)
# 	lossAvg.update_state(loss[0])
# # print(loss[0].numpy())
# replay_buffer.clear()

for epoch in range(EPOCHS):
	final_time_step, policy_state = driver.run(final_time_step, policy_state)

	# dataset = replay_buffer.as_dataset(sample_batch_size=1,num_steps=2,single_deterministic_pass=True) # do I want single_deterministic sample_batch_size=1,
	dataset = replay_buffer.as_dataset(sample_batch_size=5,num_steps=5,single_deterministic_pass=True) # do I want single_deterministic sample_batch_size=1,
	iterator = iter(dataset)
	for comp in iterator:
		trajectories, _ = comp
		loss = agent.train(experience=trajectories)
		lossAvg.update_state(loss[0])
	# print(loss[0].numpy())
	replay_buffer.clear()

	if (epoch+1) % (NUM_GAMES//10) == 0:
		print(f"done A{epoch}, {lossAvg.result().numpy()}, {env_steps.result().numpy() / num_episodes.result().numpy()}")
		lossAvg.reset_states()
		env_steps.reset()
		num_episodes.reset()

train_checkpointer = common.Checkpointer(ckpt_dir='saved_model/checkpoint2/cp',max_to_keep=1,agent=agent,policy=agent.policy,replay_buffer=replay_buffer,global_step=global_step)
train_checkpointer.save(global_step)
print("saved checkpoint")

tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save('saved_model/checkpoint2/policy')
print("saved policy")