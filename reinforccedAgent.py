import tensorflow as tf

from tf_agents.networks import network
from tf_agents.environments import tf_py_environment
from tf_agents.policies import q_policy
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver

from battleship2_env import Battleship2

environment = Battleship2()
tf_env = tf_py_environment.TFPyEnvironment(environment)

action_spec = tf_env.action_spec()
num_actions = action_spec.maximum - action_spec.minimum + 1

class QNetwork(network.Network):
	def __init__(self, input_tensor_spec, action_spec, num_actions=num_actions, name=None):
		super(QNetwork, self).__init__(
			input_tensor_spec=input_tensor_spec,
			state_spec=(),
			name=name)
		self._sub_layers = [
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(num_actions), #tweak later
		]

	def call(self, inputs, step_type=None, network_state=()):
		del step_type
		inputs = tf.cast(inputs, tf.float32)
		for layer in self._sub_layers:
			inputs = layer(inputs)
		return inputs, network_state

my_q_network = QNetwork(input_tensor_spec=tf_env.observation_spec(),action_spec=action_spec)
my_q_policy = q_policy.QPolicy(tf_env.time_step_spec(), action_spec, q_network=my_q_network)

num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [num_episodes, env_steps]

driver = dynamic_episode_driver.DynamicEpisodeDriver( # switch from episode to step
    tf_env, my_q_policy, observers, num_episodes=2)

final_time_step, policy_state = driver.run()
print('final_time_step', final_time_step)
print('Number of Steps: ', env_steps.result().numpy())
print('Number of Episodes: ', num_episodes.result().numpy())