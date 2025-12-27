import os
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """
    Actor-Critic network.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # actor (policy head)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic (value head)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class StudentAgent:
    """
    Predator agent (inference only) with separate models per adversary.
    """

    _policies = {}  # dictionary of agent_id -> ActorCritic model
    _device = torch.device("cpu")

    def __init__(self):
        self.obs_dim = 16
        self.action_dim = 5

        # load models for all adversaries if not already loaded
        if not StudentAgent._policies:
            self._load_models()

    def _load_models(self):
        """
        Load all trained models (one per adversary).
        Expects files: ppo_predator_adversary_0.pth, _1.pth, _2.pth
        """
        for i in range(3):
            model = ActorCritic(self.obs_dim, self.action_dim).to(self._device)
            model_path = os.path.join(
                os.path.dirname(__file__), f"ppo_predator_adversary_{i}.pth"
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} not found")
            state_dict = torch.load(model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.eval()
            StudentAgent._policies[f"adversary_{i}"] = model
        print("[INFO] All adversary models loaded successfully")

    def get_action(self, observation, agent_id):
        """
        Select deterministic action for a given adversary agent_id.
        """
        if isinstance(observation, dict):
            observation = observation["observation"]

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        if agent_id not in StudentAgent._policies:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        policy = StudentAgent._policies[agent_id]

        with torch.no_grad():
            action_probs, _ = policy(obs_tensor)

        return torch.argmax(action_probs, dim=-1).item()
