from herald.digest import digest_sequential
from herald.predictor import predict_amp


class HERALDEnvironment:
    # Defining the __init__ method
    def __init__(self, protein_name, protein_sequence, model, alphabet, clf):

        # Fixed for an episode
        self.protein_name = protein_name
        self.protein_sequence = protein_sequence
        self.model = model
        self.alphabet = alphabet
        self.clf = clf

        # Available choices
        self.action_space = [
            ["trypsin"],
            ["chymotrypsin"],
            ["trypsin", "chymotrypsin"],
            ["chymotrypsin", "trypsin"],
        ]

        # Conditions (optional for now, important later)
        self.ph = 7.0
        self.temperature = 37.0

        # History — what's been tried this episode
        self.tried_combinations = []
        self.rewards_history = []

        # Current state
        self.current_step = 0
        self.max_steps = len(self.action_space) * 3

    def reset(self):
        self.tried_combinations = []
        self.rewards_history = []
        self.current_step = 0

        initialstate = {
            "protein_name": self.protein_name,
            "action_space_size": len(self.action_space),
            "tried_combinations": self.tried_combinations,
            "rewards_history": self.rewards_history,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

        return initialstate

    def step(self, action):
        if action < 0 or action >= len(self.action_space):
            raise ValueError(
                f"Invalid action {action}. Must be between 0 and {len(self.action_space) - 1}"
            )
        sequence = self.protein_sequence
        enzyme_combination = self.action_space[action]
        peptides = digest_sequential(
            sequence, enzyme_combination, min_length=4, max_length=50
        )

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

        self.tried_combinations.append(enzyme_combination)
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
        pass
