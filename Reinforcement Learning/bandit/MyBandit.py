import math
import random


class Bandit:
    """Discounted-UCB bandit with decaying epsilon exploration.

    This is an algorithmic improvement over epsilon-greedy:
    - UCB explores based on uncertainty (optimism bonus).
    - Discounting makes the estimator non-stationary (the environment changes
      across the 20 runs and the bandit is NOT reset).
    - A small, decaying epsilon prevents early lock-in to a bad arm.
    """

    def __init__(self, arms, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon

        # Keep these for compatibility with the simulator/tests.
        self.frequencies = [0] * len(arms)
        self.sums = [0.0] * len(arms)
        self.expected_values = [0.0] * len(arms)

        # Discounted stats used for decision making.
        self._discount = 0.97
        self._counts = [0.0] * len(arms)
        self._reward_sums = [0.0] * len(arms)
        self._t = 0

        # UCB bonus strength.
        self._ucb_c = 1.8

        # Epsilon schedule.
        self._eps0 = 0.12
        self._eps_min = 0.01
        self._eps_decay = 4000.0

    def run(self):
        # Make sure each arm is tried at least once.
        for i in range(len(self.arms)):
            if self._counts[i] < 1e-9:
                return self.arms[i]

        self._t += 1

        eps = max(self._eps_min, self._eps0 / (1.0 + (self._t / self._eps_decay)))
        if random.random() < eps:
            return self.arms[random.randrange(0, len(self.arms))]

        total = sum(self._counts)
        log_term = math.log(total + 1.0)

        best_index = 0
        best_score = float("-inf")
        for i in range(len(self.arms)):
            mean = self._reward_sums[i] / self._counts[i]
            bonus = self._ucb_c * math.sqrt(log_term / self._counts[i])
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_index = i

        return self.arms[best_index]

    def give_feedback(self, arm, reward):
        arm_index = self.arms.index(arm)

        # Update undiscounted stats (compatibility).
        self.sums[arm_index] += reward
        self.frequencies[arm_index] += 1
        self.expected_values[arm_index] = self.sums[arm_index] / self.frequencies[arm_index]

        # Discount all arms (forgetting).
        for i in range(len(self.arms)):
            self._counts[i] *= self._discount
            self._reward_sums[i] *= self._discount

        # Add current observation.
        self._counts[arm_index] += 1.0
        self._reward_sums[arm_index] += reward
