"""
agent.py — Epsilon-greedy reinforcement learning agent for HERALD.
Defines the EpsilonGreedyAgent class, a multi-armed bandit agent that learns
which enzyme combinations produce the highest AMP probability peptides through
iterative interaction with the HERALDEnvironment.
The agent uses an epsilon-greedy exploration strategy, balancing random
exploration of untried enzyme combinations with exploitation of the best
known combination. Epsilon decays over time, shifting the agent from
exploration toward exploitation as it accumulates experience.
This is the Level 1 RL implementation for HERALD. In Level 2, the agent
policy will be replaced with a language model that proposes enzyme
combinations based on protein sequence context and experimental history,
providing a direct demonstration of reinforcement learning from verifiable
rewards (RLVR).
"""

import numpy as np


class EpsilonGreedyAgent:
    """
    Epsilon-greedy multi-armed bandit agent for enzyme combination optimization.
    Learns which enzyme combinations in the HERALDEnvironment action space
    produce the highest AMP probability rewards through trial and error.
    Maintains a running average reward estimate for each action and uses
    epsilon-greedy exploration to balance trying new combinations against
    repeating known good ones.
    The exploration rate epsilon starts high (full exploration) and decays
    toward epsilon_min (mostly exploitation) as the agent accumulates
    experience. This ensures the agent thoroughly explores the action space
    early in training before committing to the best known combination.
    Attributes:
        n_actions (int): Number of enzyme combinations in the action space.
        epsilon (float): Current exploration rate between epsilon_min and 1.0.
        epsilon_min (float): Minimum exploration rate after decay.
        epsilon_decay (float): Multiplicative decay factor applied after each step.
        action_values (numpy.ndarray): Estimated reward for each action.
        action_counts (numpy.ndarray): Number of times each action has been tried.
    """

    def __init__(self, n_actions, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        """
        Initialize the epsilon-greedy agent.
        Args:
            n_actions (int): Number of actions available in the environment.
            epsilon (float): Initial exploration rate. Default 1.0 means the
                agent explores randomly at the start of training.
            epsilon_min (float): Minimum exploration rate after decay. Default
                0.1 means the agent always has a 10% chance of exploring even
                after convergence.
            epsilon_decay (float): Multiplicative factor applied to epsilon
                after each step. Default 0.995 produces gradual decay from
                full exploration to mostly exploitation over many steps.
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # initialize value estimates and counts to zero for all actions
        self.action_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)

    def select_action(self):
        """
        Select an action using the epsilon-greedy strategy.
        With probability epsilon, selects a random action to encourage
        exploration of untried or underexplored enzyme combinations.
        Otherwise, selects the action with the highest current value
        estimate to exploit accumulated knowledge.
        Returns:
            int: Index of the selected action in the environment action space.
        """
        if np.random.rand() < self.epsilon:
            # explore — pick a random action
            return np.random.randint(self.n_actions)
        else:
            # exploit — pick the action with highest estimated value
            return np.argmax(self.action_values)

    def update(self, action, reward):
        """
        Update the agent's value estimate for an action using a running average.
        Applies the incremental update rule:
            new_value = old_value + (1 / count) * (reward - old_value)
        This is equivalent to computing a running mean without storing the
        full reward history. Each new observation nudges the estimate toward
        the true expected reward for that action.
        Also decays epsilon after each update to gradually shift the agent
        from exploration toward exploitation as training progresses.
        Args:
            action (int): Index of the action that was taken.
            reward (float): Reward received from the environment for that action.
        """
        self.action_counts[action] += 1
        self.action_values[action] = self.action_values[
            action
        ] + 1 / self.action_counts[action] * (reward - self.action_values[action])
        # decay epsilon toward epsilon_min after each step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self):
        """
        Return the action with the highest estimated value after training.
        The agent's final recommendation is the enzyme combination
        it believes produces the highest AMP probability peptides based on
        accumulated experience. Called after training is complete to extract
        the learned optimal policy.
        Returns:
            int: Index of the highest-valued action in the action space.
        """
        return np.argmax(self.action_values)

    def render(self):
        """
        Print the agent's current state for debugging and monitoring.
        Displays the current exploration rate, estimated value and try count
        for each action, and the current best action. Useful for monitoring
        learning progress during training.
        """
        print(f"Epsilon: {self.epsilon:.3f}")
        for i in range(self.n_actions):
            print(
                f"Action {i}: value={self.action_values[i]:.3f} "
                f"count={int(self.action_counts[i])}"
            )
        print(f"Best action: {self.best_action()}")
