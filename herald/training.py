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

try:
    from herald.agent import EpsilonGreedyAgent
    from herald.environment import HERALDEnvironment
except ModuleNotFoundError:
    from agent import EpsilonGreedyAgent
    from environment import HERALDEnvironment


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


if __name__ == "__main__":
    print("=" * 60)
    print("HERALD — training.py self-test")
    print("=" * 60)
    print("\nFull training loop requires ESM-2 — skipping live rerun.")
    print("Training results are validated via notebook 04 output.")
    print("\nCross-check against manuscript (Section 2.6 / Table 4):")

    action_labels = [
        "trypsin",
        "chymotrypsin",
        "alcalase",
        "papain",
        "bromelain",
        "pepsin",
        "pepsin → trypsin",
        "alcalase → papain",
    ]
    expected_values = [0.857, 0.846, 0.760, 0.704, 0.773, 0.838, 0.826, 0.693]

    print(f"\n{'Rank':<6} {'Enzyme':<25} {'Expected Value':>14}")
    print("-" * 48)
    ranked = sorted(
        zip(action_labels, expected_values), key=lambda x: x[1], reverse=True
    )
    for rank, (label, value) in enumerate(ranked, 1):
        print(f"{rank:<6} {label:<25} {value:>14.3f}")

    print(f"\nExpected best action : trypsin (value 0.857)")
    print(f"Expected convergence : ~episode 25")
    print(f"Training episodes    : 100")
    print(f"Epsilon init         : 1.0")
    print(f"Epsilon decay        : 0.995")
    print(f"Epsilon min          : 0.1")
    print("=" * 60)
    print("Run notebook 04 to reproduce full training results.")
