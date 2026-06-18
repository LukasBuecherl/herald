from mimetypes import inited

from torch._functorch.vmap import in_dims_t


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
        pass

    def render(self):
        pass
