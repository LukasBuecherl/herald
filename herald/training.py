"""
training.py — Reinforcement learning training loop for HERALD.
Provides the train() function that runs the epsilon-greedy agent through
multiple episodes of interaction with the HERALD environment to learn
which enzyme combinations produce the highest AMP probability peptides.
The training loop follows the standard RL pattern:
    1. Reset the environment at the start of each episode
    2. Agent selects an action using epsilon-greedy exploration
    3. Environment executes the action and returns a reward
    4. Agent updates its value estimates using a running average
    5. Repeat until the episode ends, then start a new episode
The agent's knowledge (action_values, action_counts, epsilon) persists
across episodes while the environment resets each time. This allows the
agent to accumulate experience across many episodes and converge on the
optimal enzyme combination.
"""

from herald.agent import EpsilonGreedyAgent
from herald.environment import HERALDEnvironment


def train(env: HERALDEnvironment, agent: EpsilonGreedyAgent, n_episodes: int):
    """
    Run the epsilon-greedy agent through multiple episodes of the HERALD environment.
    At each step the agent selects an enzyme combination, the environment
    simulates digestion and scores the resulting peptides using the ESM-2
    classifier, and the agent updates its value estimates based on the
    received reward. This process repeats for n_episodes episodes, with
    the environment resetting between episodes and the agent retaining
    its accumulated knowledge.
    The total reward per episode is recorded to produce a learning curve
    showing how the agent improves over time. A rising curve indicates
    the agent is successfully learning which enzyme combinations produce
    higher AMP probability peptides.
    Args:
        env (HERALDEnvironment): Initialized HERALD environment containing
            the target protein, ESM-2 model, and AMP classifier. Should be
            created with the target protein sequence before calling train().
        agent (EpsilonGreedyAgent): Initialized epsilon-greedy agent with
            n_actions matching the size of env.action_space. Should be
            created fresh before each training run.
        n_episodes (int): Number of episodes to train for. Each episode
            runs for env.max_steps steps. 100 episodes is sufficient for
            convergence with 4 actions.
    Returns:
        dict: Training results containing:
            episode_rewards (list[float]): Total reward per episode.
                Plot this to visualize the learning curve.
            best_action (int): Index of the highest-valued action after
                training. Use this to look up the recommended enzyme
                combination in env.action_space.
            action_values (numpy.ndarray): Final estimated reward for
                each action after training. Higher values indicate more
                consistently productive enzyme combinations.
            best_combination (list[str]): The enzyme combination the agent
                converged on as optimal. This is the primary output —
                the recommended combination for wet lab validation.
    """
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action()
            state, reward, done = env.step(action)
            agent.update(action, reward)
            total_reward += reward

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Total reward: {total_reward:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return {
        "episode_rewards": episode_rewards,
        "best_action": agent.best_action(),
        "action_values": agent.action_values,
        "best_combination": env.action_space[agent.best_action()],
    }
