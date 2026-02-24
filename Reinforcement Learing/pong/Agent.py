import numpy
import random
from collections import defaultdict


def reshape_obs(observation):
    """
    Reshapes and 'discretizes' an observation for Q-table read/write
    Make sure the state space is not too large!

    :param observation: The to-be-reshaped/discretized observation. Contains the position of the
    'players', as well as the position and movement.
    direction of the ball.
    :return: The reshaped/discretized observation
    """
    # ma-gym PongDuel observation layout (per agent), all normalized to [0, 1]:
    # [agent_row, agent_col, ball_row, ball_col, ball_dir_onehot(6)]
    obs_array = numpy.asarray(observation, dtype=float).flatten()

    agent_row = float(obs_array[0])
    ball_row = float(obs_array[2])
    ball_col = float(obs_array[3])
    ball_dir_onehot = obs_array[4:10]
    ball_dir = int(numpy.argmax(ball_dir_onehot))

    # Binning: keep it small but informative.
    agent_row_bins = 7
    ball_row_bins = 7
    ball_col_bins = 8

    def bin01(x, bins):
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        idx = int(x * bins)
        return min(bins - 1, idx)

    agent_row_b = bin01(agent_row, agent_row_bins)
    ball_row_b = bin01(ball_row, ball_row_bins)
    ball_col_b = bin01(ball_col, ball_col_bins)

    return (agent_row_b, ball_row_b, ball_col_b, ball_dir)


class Agent:
    """
    Skeleton q-learner agent that the students have to implement
    """

    def __init__(
            self, id, actions_n, obs_space_shape,
            gamma=0.95,  # Discount factor for future rewards
            epsilon=1.0,  # Initial exploration rate
            min_epsilon=0.01,  # Minimum exploration rate
            epsilon_decay=0.995,  # Decay rate per episode
            alpha=0.2  # Learning rate
    ):
        """
        Initiates the agent

        :param id: The agent's id in the game environment
        :param actions_n: The id of actions in the agent's action space
        :param obs_space_shape: The shape of the agents observation space
        :param gamma: Depreciation factor for expected future rewards
        :param epsilon: The initial/current exploration rate
        :param min_epsilon: The minimal/final exploration rate
        :param epsilon_decay: The rate of epsilon/exploration decay
        :param alpha: The learning rate
        """
        self.id = id
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_n = actions_n
        self.obs_space_shape = obs_space_shape
        self.alpha = alpha
        self.q = defaultdict(lambda: numpy.zeros(self.actions_n))
        self.episode_count = 0
        self._epsilon_step_decay = 0.99995

    def determine_action_probabilities(self, observation):
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a numpy array of length of the action space.
        Uses epsilon-greedy strategy: with probability epsilon, pick random action;
        otherwise pick the action with highest Q-value.
        
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a numpy
        array of length of the action space.
        """
        action_probabilities = numpy.zeros(self.actions_n)
        
        # Get Q-values for current state
        state = reshape_obs(observation[self.id])
        q_values = self.q[state]
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_probabilities = numpy.ones(self.actions_n) / self.actions_n
        else:
            # Random tie-break among best actions (prevents a hard NOOP bias)
            best_value = numpy.max(q_values)
            best_actions = numpy.flatnonzero(q_values == best_value)
            best_action = int(random.choice(best_actions.tolist()))
            action_probabilities[best_action] = 1.0
        
        return action_probabilities

    def act(self, observation):
        """
        Determines and action, given the current observation.
        :param observation: the agent's current observation of the state of
        the world
        :return: the agent's action
        """
        action_probabilities = self.determine_action_probabilities(observation)
        action = numpy.random.choice(self.actions_n, p=action_probabilities)
        return action

    def update_history(
            self, observation, action, reward, new_observation
    ):
        """
        Updates the agent's Q-table using Q-learning update rule

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        :return:
        """
        # Get current state and new state
        state = reshape_obs(observation[self.id])
        new_state = reshape_obs(new_observation[self.id])
        
        # Treat scoring events as terminal (ball resets after a point)
        if reward != 0:
            td_target = reward
        else:
            td_target = reward + self.gamma * numpy.max(self.q[new_state])
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta
        
        # Decay epsilon over time (pong.py doesn't currently call into Agent per-episode)
        self.epsilon = max(self.min_epsilon, self.epsilon * self._epsilon_step_decay)

