"""
environment.py — Reinforcement learning environment for HERALD.
Defines the HERALDEnvironment class, a reinforcement learning environment
that simulates enzymatic hydrolysis of whey proteins and scores the resulting
peptide fragments using an ESM-2-based AMP classifier.
The environment follows the standard RL interface with reset(), step(), and
render() methods. At each step the agent selects an enzyme combination from
the action space, the environment simulates digestion and scores the resulting
peptides, and returns a reward based on the predicted AMP probability of the
peptide profile.
The reward signal is currently based on ESM-2 classifier probabilities as a
proxy for experimental antimicrobial activity. In future iterations, wet lab
MIC values will be incorporated to recalibrate the reward function.
"""

from herald.digestion import digest_sequential
from herald.enzymes import get_conditions
from herald.predictor import predict_amp


class HERALDEnvironment:
    """
    Reinforcement learning environment for enzyme combination optimization.
    Models the enzymatic hydrolysis of a whey protein as a sequential
    decision-making problem. At each step the agent selects an enzyme
    combination, the environment simulates digestion, scores the resulting
    peptides using an ESM-2 AMP classifier, and returns a reward signal.s
    The action space consists of single and sequential enzyme combinations
    drawn from food-grade proteases defined in enzymes.py. The reward
    combines average and maximum AMP probability across all peptides produced
    by the selected combination.

    Attributes:
        protein_name (str): Name of the target whey protein.
        protein_sequence (str): Amino acid sequence of the target protein.
        model: Pretrained ESM-2 model for sequence embedding.
        alphabet: ESM-2 alphabet object used for tokenization.
        clf: Fitted logistic regression AMP classifier.
        action_space (list): List of enzyme combinations available to the agent.
        ph (float): Hydrolysis pH — fixed for now, part of action space later.
        temperature (float): Hydrolysis temperature in Celsius — fixed for now.
        tried_combinations (list): History of enzyme combinations tried this episode.
        rewards_history (list): History of rewards received this episode.
        current_step (int): Number of steps taken in the current episode.
        max_steps (int): Maximum steps per episode before termination.
    """

    def __init__(self, protein_name, protein_sequence, model, alphabet, clf):
        """
        Initialize the HERALD RL environment.
        Args:
            protein_name (str): Name of the target whey protein.
            protein_sequence (str): Amino acid sequence of the target protein.
            model: Pretrained ESM-2 model for sequence embedding.
            alphabet: ESM-2 alphabet object used for tokenization.
            clf: Fitted logistic regression AMP classifier.
        """
        # fixed for the lifetime of this environment
        self.protein_name = protein_name
        self.protein_sequence = protein_sequence
        self.model = model
        self.alphabet = alphabet
        self.clf = clf

        # action space — single and sequential enzyme combinations
        ENZYME_COMBINATIONS = [
            {"enzymes": ["trypsin"], "label": "trypsin"},
            {"enzymes": ["chymotrypsin"], "label": "chymotrypsin"},
            {"enzymes": ["alcalase"], "label": "alcalase"},
            {"enzymes": ["papain"], "label": "papain"},
            {"enzymes": ["bromelain"], "label": "bromelain"},
            {"enzymes": ["pepsin"], "label": "pepsin"},
            {"enzymes": ["pepsin", "trypsin"], "label": "pepsin → trypsin"},
            {"enzymes": ["alcalase", "papain"], "label": "alcalase → papain"},
        ]

        self.action_space = []
        for combo in ENZYME_COMBINATIONS:
            ph, temp = get_conditions(combo["enzymes"])
            self.action_space.append(
                {
                    "enzymes": combo["enzymes"],
                    "label": combo["label"],
                    "ph": ph,
                    "temp": temp,
                }
            )

        # episode state — reset at the start of each episode
        self.tried_combinations = []
        self.rewards_history = []
        self.current_step = 0
        self.max_steps = len(self.action_space) * 3

    def reset(self):
        """
        Reset the environment to its initial state for a new episode.
        Clears episode history and resets the step counter. Should be called
        at the start of each new training episode.
        Returns:
            dict: Initial state containing protein name, action space size,
            empty history, and episode step information.
        """
        self.tried_combinations = []
        self.rewards_history = []
        self.current_step = 0
        return {
            "protein_name": self.protein_name,
            "action_space_size": len(self.action_space),
            "tried_combinations": self.tried_combinations,
            "rewards_history": self.rewards_history,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

    def step(self, action):
        """
        Execute one step in the environment by applying an enzyme combination.
        The agent selects an action index corresponding to an enzyme combination
        in the action space. The environment simulates sequential digestion of
        the target protein, scores the resulting peptides using the ESM-2 AMP
        classifier, and returns a weighted reward combining average and maximum
        AMP probability.
        The reward formula is:
            reward = 0.4 * avg_probability + 0.6 * max_probability
        This weights the best peptide produced more heavily than the average,
        rewarding combinations that generate at least one strong AMP candidate
        while still valuing overall combination quality.
        Args:
            action (int): Index into action_space selecting an enzyme combination.
        Returns:
            tuple: (state, reward, done) where:
                state (dict): Updated environment state after this step.
                reward (float): AMP probability-based reward for this action.
                done (bool): True if the episode has reached max_steps.
        Raises:
            ValueError: If action is outside the valid action space range.
        """
        if action < 0 or action >= len(self.action_space):
            raise ValueError(
                f"Invalid action {action}. "
                f"Must be between 0 and {len(self.action_space) - 1}"
            )

        sequence = self.protein_sequence
        enzyme_combination = self.action_space[action]["enzymes"]

        # simulate enzymatic digestion with the selected combination
        peptides = digest_sequential(
            sequence, enzyme_combination, min_length=4, max_length=50
        )

        # calculate reward from AMP probabilities of resulting peptides
        if len(peptides) == 0:
            reward = 0.0
        else:
            probabilities = [
                predict_amp(peptide, self.model, self.alphabet, self.clf)[1]
                for peptide in peptides
            ]
            avg_probability = sum(probabilities) / len(probabilities)
            max_probability = max(probabilities)
            reward = 0.4 * avg_probability + 0.6 * max_probability

        # update episode history
        self.tried_combinations.append(self.action_space[action])
        self.rewards_history.append(reward)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        state = {
            "protein_name": self.protein_name,
            "action_space_size": len(self.action_space),
            "tried_combinations": self.tried_combinations,
            "rewards_history": self.rewards_history,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

        return state, reward, done

    def render(self):
        """
        Print the current environment state for debugging and monitoring.
        Displays the current step, last action taken, last reward received,
        best reward seen so far this episode, and full reward history.
        Safe to call at any point including before any steps are taken.
        """
        if len(self.rewards_history) == 0:
            print("No steps taken yet.")
            return
        print(
            f"Step {self.current_step}/{self.max_steps} | Protein: {self.protein_name}"
        )
        print(
            f"Last action: {self.tried_combinations[-1]['label']} | Reward: {self.rewards_history[-1]:.3f}"
        )
        print(f"Best reward so far: {max(self.rewards_history):.3f}")
        print(f"All rewards: {[round(r, 3) for r in self.rewards_history]}")
