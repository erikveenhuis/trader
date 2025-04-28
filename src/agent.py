import torch
import torch.optim as optim
import logging
import numpy as np
import time
import random
from collections import deque
from .buffer import PrioritizedReplayBuffer
from .model import RainbowNetwork
import os  # Added for save/load path handling
from .utils.utils import set_seeds
import yaml  # Added for config load/save
from .constants import ACCOUNT_STATE_DIM # Import constant

logger = logging.getLogger("Agent")

# --- Start: Rainbow DQN Agent ---
class RainbowDQNAgent:
    """
    Rainbow DQN Agent incorporating:
    - Distributional RL (C51)
    - Prioritized Experience Replay (PER)
    - Dueling Networks (Implicit in RainbowNetwork)
    - Multi-step Returns
    - Double Q-Learning
    - Noisy Nets for exploration
    """

    def __init__(self, config: dict, device: str = "cuda"):
        """
        Initializes the Rainbow DQN Agent.

        Args:
            config (dict): A dictionary containing all hyperparameters and network settings.
                           Expected keys: seed, gamma, lr, replay_buffer_size, batch_size,
                           target_update_freq, num_atoms, v_min, v_max, alpha, beta_start,
                           beta_frames, n_steps, window_size, n_features, hidden_dim,
                           num_actions, grad_clip_norm, debug (optional).
            device (str): The device to run the agent on ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device
        # Use direct access for mandatory parameters
        self.seed = config["seed"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.n_steps = config["n_steps"]
        self.num_atoms = config["num_atoms"]
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]
        self.num_actions = config["num_actions"]
        self.window_size = config["window_size"]
        self.n_features = config["n_features"]
        self.hidden_dim = config["hidden_dim"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.alpha = config["alpha"]
        self.beta_start = config["beta_start"]
        self.beta_frames = config["beta_frames"]
        self.grad_clip_norm = config["grad_clip_norm"]
        # Optional flags can still use .get()
        self.debug_mode = config.get("debug", False)

        # Setup seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
            logger.info("CUDA seed set.")
        elif self.device == "cuda":
            logger.warning("CUDA device requested but not available. Using CPU.")
            self.device = "cpu"

        logger.info(f"Initializing RainbowDQNAgent on {self.device}")
        logger.info(f"Config: {config}")  # Log the entire config

        # Distributional RL setup
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(
            self.device
        )
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Initialize Networks
        # Pass the agent's config dictionary and device directly
        self.network = RainbowNetwork(config=self.config, device=self.device).to(
            self.device
        )
        self.target_network = RainbowNetwork(config=self.config, device=self.device).to(
            self.device
        )

        # # Check if PyTorch version >= 2.0 to use torch.compile
        # if int(torch.__version__.split('.')[0]) >= 2:
        #     logger.info("Applying torch.compile to network and target_network.")
        #     # Add error handling in case compile fails on specific setups
        #     try:
        #         # Try the default compilation mode first
        #         self.network = torch.compile(self.network)
        #         self.target_network = torch.compile(self.target_network)
        #         logger.info("torch.compile applied successfully with default mode.")
        #     except ImportError as imp_err: # Catch potential import errors if compile isn't fully set up
        #          logger.warning(f"torch.compile skipped due to potential import issue: {imp_err}. Proceeding without compilation.")
        #     except Exception as e:
        #          # Check if it's the TritonMissing error specifically
        #          # Check class name string as direct import might fail if torch._inductor isn't available
        #          if "TritonMissing" in str(e.__class__):
        #              logger.warning(f"torch.compile failed as Triton backend is not available (common on non-Linux/CUDA setups): {e}. Proceeding without compilation.")
        #          else:
        #              logger.warning(f"torch.compile failed with an unexpected error: {e}. Proceeding without compilation.")
        # else:
        #     logger.warning("torch version < 2.0, torch.compile not available.")

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        logger.info("Rainbow networks and optimizer created.")
        logger.info(
            f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}"
        )

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            self.replay_buffer_size,
            self.alpha,
            self.beta_start,
            self.beta_frames,
        )
        # For N-step returns
        self.n_step_buffer = deque(maxlen=self.n_steps)

        self.training_mode = True  # Start in training mode by default
        self.total_steps = (
            0  # Track total steps for target network updates and beta annealing
        )

    def select_action(self, obs):
        """Selects action based on the current Q-value estimates using Noisy Nets."""
        assert isinstance(obs, dict), "Observation must be a dictionary"
        assert (
            "market_data" in obs and "account_state" in obs
        ), "Observation missing required keys"
        assert isinstance(
            obs["market_data"], np.ndarray
        ), "obs['market_data'] must be a numpy array"
        assert isinstance(
            obs["account_state"], np.ndarray
        ), "obs['account_state'] must be a numpy array"
        # Check shapes (before adding batch dimension)
        assert obs["market_data"].shape == (
            self.window_size,
            self.n_features,
        ), f"Input market_data shape mismatch. Expected {(self.window_size, self.n_features)}, got {obs['market_data'].shape}"
        assert obs["account_state"].shape == (
            ACCOUNT_STATE_DIM,
        ), f"Input account_state shape mismatch. Expected ({ACCOUNT_STATE_DIM},), got {obs['account_state'].shape}"

        # Convert observation to tensors
        market_data = torch.FloatTensor(obs["market_data"]).unsqueeze(0).to(self.device)
        account_state = (
            torch.FloatTensor(obs["account_state"]).unsqueeze(0).to(self.device)
        )
        assert market_data.shape == (
            1,
            self.window_size,
            self.n_features,
        ), "Tensor market_data shape mismatch"
        assert account_state.shape == (
            1,
            ACCOUNT_STATE_DIM,
        ), "Tensor account_state shape mismatch"

        # Select action using the online network (with Noisy Layers for exploration)
        self.network.eval()  # Ensure eval mode for action selection if using dropout/batchnorm (though NoisyNets handle exploration)
        with torch.no_grad():
            q_values = self.network.get_q_values(market_data, account_state)
            assert q_values.shape == (
                1,
                self.num_actions,
            ), f"Q-values shape mismatch. Expected (1, {self.num_actions}), got {q_values.shape}"
            action = (
                q_values.argmax().item()
            )  # Choose action with highest expected Q-value
            assert isinstance(
                action, int
            ), f"Selected action is not an integer: {action}"
            assert (
                0 <= action < self.num_actions
            ), f"Selected action ({action}) is out of bounds [0, {self.num_actions})"
        # Switch back to train mode if necessary (depends if eval() affects Noisy Layers - typically it doesn't disable noise generation)
        if self.training_mode:
            self.network.train()

        return action

    def _get_n_step_info(self):
        """
        Calculates the n-step return G_t^(n) and identifies the state s_{t+n}
        and done flag d_{t+n} from the transition n steps later.
        Uses the n-step buffer which contains (s_k, a_k, r_{k+1}, s_{k+1}, d_{k+1}).

        Returns:
            tuple: Contains:
                - state_t (tuple): (market_data_t, account_state_t) from the start of the n steps.
                - action_t (int): Action taken at state_t.
                - n_step_reward (float): The accumulated discounted n-step return.
                - next_state_tn (tuple): (market_data_{t+n}, account_state_{t+n}) observed n steps later.
                - done_tn (bool): Done flag from n steps later.
        """
        assert (
            len(self.n_step_buffer) == self.n_steps
        ), "N-step buffer size mismatch for calculation"

        # State and action from the *first* transition in the buffer (t)
        market_data_t, account_state_t, action_t, _, _, _, _ = self.n_step_buffer[0]
        state_t = (market_data_t, account_state_t)

        # Next state and done flag from the *last* transition in the buffer (t+n)
        _, _, _, _, next_market_tn, next_account_tn, done_tn = self.n_step_buffer[-1]
        next_state_tn = (next_market_tn, next_account_tn)

        # Calculate cumulative discounted reward G_t^(n)
        n_step_reward = 0.0
        discount = 1.0
        for i in range(self.n_steps):
            # r_{t+i+1}, d_{t+i+1}
            _, _, _, reward_iplus1, _, _, done_iplus1 = self.n_step_buffer[i]
            n_step_reward += discount * reward_iplus1
            discount *= self.gamma
            if done_iplus1:
                # If termination occurs at step t+i+1 (where i < n-1),
                # the n-step return calculation stops here.
                # The next_state and done used for the Bellman target are correctly
                # s_{t+n} and d_{t+n} from the last transition.
                break

        # --- Start: Assert return types and shapes ---
        assert (
            isinstance(state_t, tuple) and len(state_t) == 2
        ), "state_t is not a 2-tuple"
        assert isinstance(state_t[0], np.ndarray) and state_t[0].shape == (
            self.window_size,
            self.n_features,
        ), "state_t[0] (market_data) has wrong type/shape"
        assert isinstance(state_t[1], np.ndarray) and state_t[1].shape == (
            ACCOUNT_STATE_DIM,
        ), "state_t[1] (account_state) has wrong type/shape"
        assert isinstance(action_t, (int, np.integer)), "action_t is not an integer"
        assert isinstance(
            n_step_reward, (float, np.float32, np.float64)
        ), "n_step_reward is not a float"
        assert (
            isinstance(next_state_tn, tuple) and len(next_state_tn) == 2
        ), "next_state_tn is not a 2-tuple"
        assert isinstance(next_state_tn[0], np.ndarray) and next_state_tn[0].shape == (
            self.window_size,
            self.n_features,
        ), "next_state_tn[0] (market_data) has wrong type/shape"
        assert isinstance(next_state_tn[1], np.ndarray) and next_state_tn[1].shape == (
            ACCOUNT_STATE_DIM,
        ), "next_state_tn[1] (account_state) has wrong type/shape"
        assert isinstance(done_tn, (bool, np.bool_)), "done_tn is not a boolean"
        # --- End: Assert return types and shapes ---

        return state_t, action_t, n_step_reward, next_state_tn, done_tn

    def store_transition(self, obs, action, reward, next_obs, done):
        """Stores experience in N-step buffer and potentially transfers to PER."""
        # --- Start: Assert input types and shapes for store_transition ---
        assert (
            isinstance(obs, dict) and "market_data" in obs and "account_state" in obs
        ), "Invalid current observation format"
        assert (
            isinstance(next_obs, dict)
            and "market_data" in next_obs
            and "account_state" in next_obs
        ), "Invalid next observation format"
        assert isinstance(obs["market_data"], np.ndarray) and obs[
            "market_data"
        ].shape == (
            self.window_size,
            self.n_features,
        ), f"Invalid obs market data shape {obs['market_data'].shape}"
        assert isinstance(obs["account_state"], np.ndarray) and obs[
            "account_state"
        ].shape == (
            ACCOUNT_STATE_DIM,
        ), f"Invalid obs account state shape {obs['account_state'].shape}"
        assert isinstance(next_obs["market_data"], np.ndarray) and next_obs[
            "market_data"
        ].shape == (
            self.window_size,
            self.n_features,
        ), f"Invalid next_obs market data shape {next_obs['market_data'].shape}"
        assert isinstance(next_obs["account_state"], np.ndarray) and next_obs[
            "account_state"
        ].shape == (
            ACCOUNT_STATE_DIM,
        ), f"Invalid next_obs account state shape {next_obs['account_state'].shape}"
        assert isinstance(action, (int, np.integer)), "Action must be an integer"
        assert isinstance(
            reward, (float, np.float32, np.float64)
        ), "Reward must be a float"
        assert isinstance(done, (bool, np.bool_)), "Done flag must be boolean"
        # --- End: Assert input types and shapes ---

        # Store raw single-step transition data needed for n-step calculation
        # (s_k, a_k, r_{k+1}, s_{k+1}_market, s_{k+1}_account, d_{k+1})
        transition = (
            obs["market_data"],
            obs["account_state"],
            action,
            reward,
            next_obs["market_data"],
            next_obs["account_state"],
            done,
        )
        self.n_step_buffer.append(transition)

        # If buffer has enough steps, calculate N-step return and store in PER
        if len(self.n_step_buffer) >= self.n_steps:
            state_t, action_t, n_step_reward, next_state_tn, done_tn = (
                self._get_n_step_info()
            )
            market_data_t, account_state_t = state_t
            next_market_tn, next_account_tn = next_state_tn

            # --- Start: Assert types before storing in PER buffer ---
            assert isinstance(market_data_t, np.ndarray) and market_data_t.shape == (
                self.window_size,
                self.n_features,
            ), "Invalid market_data_t for PER store"
            assert isinstance(
                account_state_t, np.ndarray
            ) and account_state_t.shape == (
                ACCOUNT_STATE_DIM,
            ), "Invalid account_state_t for PER store"
            assert isinstance(
                action_t, (int, np.integer)
            ), "Invalid action_t for PER store"
            assert isinstance(
                n_step_reward, (float, np.float32, np.float64)
            ), "Invalid n_step_reward for PER store"
            assert isinstance(next_market_tn, np.ndarray) and next_market_tn.shape == (
                self.window_size,
                self.n_features,
            ), "Invalid next_market_tn for PER store"
            assert isinstance(
                next_account_tn, np.ndarray
            ) and next_account_tn.shape == (
                ACCOUNT_STATE_DIM,
            ), "Invalid next_account_tn for PER store"
            assert isinstance(
                done_tn, (bool, np.bool_)
            ), "Invalid done_tn flag for PER store"
            # --- End: Assert types before storing ---

            # Store the calculated N-step transition in the main prioritized buffer
            # Format: (s_t, a_t, G_t^(n), s_{t+n}, d_{t+n})
            self.buffer.store(
                market_data_t,
                account_state_t,
                action_t,
                n_step_reward,
                next_market_tn,
                next_account_tn,
                done_tn,
            )

    def _project_target_distribution(
        self, next_market_data_batch, next_account_state_batch, rewards, dones
    ):
        """
        Computes the projected target distribution for the C51 algorithm.
        Applies the Bellman update for n-steps and projects the resulting
        distribution onto the fixed support atoms.

        Args:
            next_market_data_batch (torch.Tensor): Batch of next market data states (s_{t+n}).
            next_account_state_batch (torch.Tensor): Batch of next account states (s_{t+n}).
            rewards (torch.Tensor): Batch of n-step rewards (G_t^(n)), shape [B, 1].
            dones (torch.Tensor): Batch of done flags (d_{t+n}), shape [B, 1].

        Returns:
            torch.Tensor: The projected target distribution (m) with shape [batch_size, num_atoms].
        """
        with torch.no_grad():
            # Double DQN: Use online network to select best next action's index at state s_{t+n}
            next_q_values = self.network.get_q_values(
                next_market_data_batch, next_account_state_batch
            )
            assert next_q_values.shape == (
                self.batch_size,
                self.num_actions,
            ), "Next Q-values shape mismatch"
            next_actions = next_q_values.argmax(dim=1)  # [batch_size]
            assert next_actions.shape == (
                self.batch_size,
            ), "Next actions shape mismatch"

            # Get next state's distribution Z(s_{t+n}, a*) from target network for selected actions a*
            next_log_dist = self.target_network(
                next_market_data_batch, next_account_state_batch
            )  # [B, num_actions, num_atoms]
            assert next_log_dist.shape == (
                self.batch_size,
                self.num_actions,
                self.num_atoms,
            ), "Next log distribution shape mismatch"
            assert (
                next_actions.max() < self.num_actions and next_actions.min() >= 0
            ), "Invalid next_action indices"

            # Get the probability distribution for the chosen actions: p(s_{t+n}, a*)
            next_dist = torch.exp(
                next_log_dist[range(self.batch_size), next_actions]
            )  # [B, num_atoms]
            assert next_dist.shape == (
                self.batch_size,
                self.num_atoms,
            ), "Next distribution shape mismatch"

            # Compute the projected Bellman target T_z = G_t^(n) + gamma^n * Z(s_{t+n}, a*)
            # Rewards are [B, 1], dones are [B, 1], support is [num_atoms]
            # Broadcasting applies correctly.
            Tz = (
                rewards + (1 - dones) * (self.gamma**self.n_steps) * self.support
            )  # [B, num_atoms]
            assert Tz.shape == (
                self.batch_size,
                self.num_atoms,
            ), f"Projected Tz shape mismatch: {Tz.shape}"
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            # Compute projection indices and weights
            b = (
                Tz - self.v_min
            ) / self.delta_z  # Normalized position on support axis [B, num_atoms]
            assert b.shape == (
                self.batch_size,
                self.num_atoms,
            ), f"Projection 'b' shape mismatch: {b.shape}"
            lower_atom_idx = b.floor().long()
            u = b.ceil().long()  # Upper atom index
            # Fix disappearing probability mass when l = b = u (b is int)
            lower_atom_idx[(u > 0) * (lower_atom_idx == u)] -= 1
            u[(lower_atom_idx < (self.num_atoms - 1)) * (lower_atom_idx == u)] += 1

            # Distribute probability
            m = torch.zeros_like(next_dist)
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.num_atoms, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.num_atoms)
                .to(self.device)
            )

            # Ensure indices are within bounds [0, num_atoms - 1]
            lower_atom_idx = lower_atom_idx.clamp(0, self.num_atoms - 1)
            u = u.clamp(0, self.num_atoms - 1)

            # Debugging shapes right before indexing
            # logger.debug(f"Shapes before indexing: m={m.shape}, offset={offset.shape}, l={l.shape}, u={u.shape}, next_dist={next_dist.shape}, b={b.shape}")
            # logger.debug(f"Indices: l max={l.max()}, min={l.min()}; u max={u.max()}, min={u.min()}")

            m.view(-1).index_add_(
                0,
                (lower_atom_idx + offset).view(-1),
                (next_dist * (u.float() - b)).view(-1),
            )
            m.view(-1).index_add_(
                0,
                (u + offset).view(-1),
                (next_dist * (b - lower_atom_idx.float())).view(-1),
            )

            # Optional: Check if target distribution sums to 1 (approximately)
            # This check can be expensive, use cautiously or only in debug mode
            if self.debug_mode:
                sums = m.sum(dim=1)
                if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
                    logger.warning(
                        f"Target distribution M does not sum to 1. Sums: {sums}. Min sum: {sums.min()}, Max sum: {sums.max()}"
                    )
                    # It might not sum *exactly* to 1 due to floating point, clamping, and edge cases.
                    # A small tolerance is usually acceptable.

            return m

    def _compute_loss(self, batch, weights):
        """Computes the C51 loss using PER weights."""
        (
            market_data,
            account_state,
            actions,
            rewards,
            next_market_data,
            next_account_state,
            dones,
        ) = batch
        # --- Start: Assert batch shapes and types ---
        assert market_data.shape == (
            self.batch_size,
            self.window_size,
            self.n_features,
        ), "Batch market_data shape mismatch"
        assert account_state.shape == (
            self.batch_size,
            ACCOUNT_STATE_DIM,
        ), "Batch account_state shape mismatch"
        assert actions.shape == (self.batch_size,), "Batch actions shape mismatch"
        assert rewards.shape == (self.batch_size,), "Batch rewards shape mismatch"
        assert next_market_data.shape == (
            self.batch_size,
            self.window_size,
            self.n_features,
        ), "Batch next_market_data shape mismatch"
        assert next_account_state.shape == (
            self.batch_size,
            ACCOUNT_STATE_DIM,
        ), "Batch next_account_state shape mismatch"
        assert dones.shape == (self.batch_size,), "Batch dones shape mismatch"
        assert weights.shape == (self.batch_size,), "Batch weights shape mismatch"
        # --- End: Assert batch shapes and types ---

        # Convert numpy arrays from buffer to tensors
        market_data_batch = torch.FloatTensor(market_data).to(self.device)
        account_state_batch = torch.FloatTensor(account_state).to(self.device)
        next_market_data_batch = torch.FloatTensor(next_market_data).to(self.device)
        next_account_state_batch = torch.FloatTensor(next_account_state).to(self.device)
        actions_batch = torch.LongTensor(actions).to(self.device)  # Action indices [B]
        rewards_batch = (
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        )  # [B, 1]
        dones_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # [B, 1]
        weights_batch = (
            torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        )  # [B, 1]

        # --- Start: Assert tensor shapes after conversion ---
        assert market_data_batch.shape == (
            self.batch_size,
            self.window_size,
            self.n_features,
        ), "Tensor market_data_batch shape mismatch"
        assert account_state_batch.shape == (
            self.batch_size,
            ACCOUNT_STATE_DIM,
        ), "Tensor account_state_batch shape mismatch"
        assert next_market_data_batch.shape == (
            self.batch_size,
            self.window_size,
            self.n_features,
        ), "Tensor next_market_data_batch shape mismatch"
        assert next_account_state_batch.shape == (
            self.batch_size,
            ACCOUNT_STATE_DIM,
        ), "Tensor next_account_state_batch shape mismatch"
        assert actions_batch.shape == (
            self.batch_size,
        ), "Tensor actions_batch shape mismatch"
        assert rewards_batch.shape == (
            self.batch_size,
            1,
        ), "Tensor rewards_batch shape mismatch"
        assert dones_batch.shape == (
            self.batch_size,
            1,
        ), "Tensor dones_batch shape mismatch"
        assert weights_batch.shape == (
            self.batch_size,
            1,
        ), "Tensor weights_batch shape mismatch"
        # --- End: Assert tensor shapes ---

        # --- Calculate Target Distribution (m) --- #
        # This is the projected distribution for the Bellman target Z(s_t, a_t)
        target_distribution = self._project_target_distribution(
            next_market_data_batch, next_account_state_batch, rewards_batch, dones_batch
        )
        assert target_distribution.shape == (
            self.batch_size,
            self.num_atoms,
        ), "Target distribution shape mismatch"
        # ---------------------------------------- #

        # --- Calculate Online Distribution and Loss --- #
        # Get log probabilities Z(s_t, a) from the online network
        log_ps = self.network(
            market_data_batch, account_state_batch
        )  # [B, num_actions, num_atoms]
        assert log_ps.shape == (
            self.batch_size,
            self.num_actions,
            self.num_atoms,
        ), "Online log_ps shape mismatch"

        # Gather the log-probabilities for the actions actually taken: log Z(s_t, a_t)
        # We need to select the log probabilities corresponding to actions_batch
        actions_indices = actions_batch.view(self.batch_size, 1, 1).expand(
            self.batch_size, 1, self.num_atoms
        )
        log_ps_a = log_ps.gather(1, actions_indices).squeeze(1)  # [B, num_atoms]
        assert log_ps_a.shape == (
            self.batch_size,
            self.num_atoms,
        ), "Online log_ps_a shape mismatch"

        # Calculate cross-entropy loss between target and online distributions
        # Loss = -sum_i [ target_distribution_i * log(online_distribution_i) ]
        # Target distribution is detached as it acts as the label.
        loss_elementwise = -(target_distribution.detach() * log_ps_a).sum(dim=1)  # [B]
        assert loss_elementwise.shape == (
            self.batch_size,
        ), "Per-sample loss shape mismatch"

        # Apply Importance Sampling weights and calculate mean loss
        loss = (loss_elementwise * weights_batch.squeeze(1).detach()).mean()  # Scalar
        assert loss.ndim == 0, "Final loss is not a scalar"
        assert torch.isfinite(
            loss
        ), f"Loss calculation resulted in NaN or Inf: {loss.item()}"
        # ----------------------------------------- #

        # --- Calculate TD errors for PER update --- #
        # TD error is | E[Target Distribution] - E[Online Distribution for a_t] |
        # Calculate expected Q-values E[Z(s_t, a_t)] from the online distribution for the action taken
        q_values_online = (torch.exp(log_ps_a) * self.support.unsqueeze(0)).sum(
            dim=1
        )  # [B]
        # Calculate expected target Q-values E[Projected Target Distribution]
        q_values_target = (target_distribution * self.support.unsqueeze(0)).sum(
            dim=1
        )  # [B]
        # TD error = |Target Q - Online Q|
        td_errors_tensor = (q_values_target.detach() - q_values_online.detach()).abs()
        assert td_errors_tensor.shape == (
            self.batch_size,
        ), "TD errors tensor shape mismatch"
        assert torch.isfinite(
            td_errors_tensor
        ).all(), "NaN or Inf found in TD errors tensor"
        # ----------------------------------------- #

        return loss, td_errors_tensor

    def learn(self):
        """Samples from PER, computes loss, updates network, priorities, and target network."""
        if len(self.buffer) < self.batch_size:
            return None  # Not enough samples to learn yet

        # Sample batch from PER
        # Update beta (annealing) based on total steps *before* sampling
        self.buffer.update_beta(self.total_steps)
        beta = self.buffer.beta  # Get current beta for logging
        batch_tuple, tree_indices, weights = self.buffer.sample(
            self.batch_size
        )  # Sample returns tree_indices now

        if batch_tuple is None:
            logger.warning("PER sample returned None.")
            return None  # Should not happen if buffer size check passed, but safeguard

        # Compute loss and TD errors (TD errors are tensors)
        loss, td_errors_tensor = self._compute_loss(batch_tuple, weights)
        assert (
            isinstance(loss, torch.Tensor) and loss.ndim == 0
        ), "Loss from _compute_loss is not a scalar tensor"
        assert isinstance(
            td_errors_tensor, torch.Tensor
        ) and td_errors_tensor.shape == (
            self.batch_size,
        ), "TD errors tensor from _compute_loss has wrong shape/type"

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if self.debug_mode:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.error(
                        f"NaN or Inf detected in gradients BEFORE clipping for parameter: {p.shape}"
                    )

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), max_norm=self.grad_clip_norm
        )

        if self.debug_mode:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.error(
                        f"NaN or Inf detected in gradients AFTER clipping (Max Norm: {self.grad_clip_norm}) for parameter: {p.shape}"
                    )

        self.optimizer.step()

        # Update priorities in PER using the TD errors (ensure they are positive)
        # Add a small epsilon to prevent priorities of 0
        priorities = td_errors_tensor.cpu().numpy() + 1e-6
        assert np.isfinite(
            priorities
        ).all(), "Non-finite priorities calculated for PER update"
        self.buffer.update_priorities(
            tree_indices, td_errors_tensor
        )  # Pass tree_indices and the original tensor

        # Reset noise in Noisy Linear layers (important!)
        self.network.reset_noise()
        self.target_network.reset_noise()

        # Increment step counter and update target network periodically
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
            logger.info(f"Step {self.total_steps}: Target network updated.")

        loss_item = loss.item()
        assert (
            isinstance(loss_item, float)
            and not np.isnan(loss_item)
            and not np.isinf(loss_item)
        ), "Final loss item is not a valid float"

        # Log loss and PER beta
        logger.debug(
            f"Step: {self.total_steps}, Loss: {loss_item:.4f}, PER Beta: {beta:.4f}"
        )

        return loss_item  # Return loss for external logging/monitoring

    def _update_target_network(self):
        """Copies weights from online network to target network."""
        self.target_network.load_state_dict(self.network.state_dict())
        logger.debug("Target network weights updated.")

    def save_model(self, path_prefix):
        """Saves the agent's state to separate files: network, optimizer, and misc config."""
        if self.network is None or self.optimizer is None:
            logger.error(
                "Attempted to save model, but network or optimizer not initialized."
            )
            return

        # Ensure directory exists
        directory = os.path.dirname(path_prefix)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory for saving model: {directory}")
            except OSError as e:
                logger.error(
                    f"Could not create directory {directory} for saving model: {e}"
                )
                return # Cannot save if directory creation fails

        network_path = f"{path_prefix}_network.pth"
        optimizer_path = f"{path_prefix}_optimizer.pth"
        misc_path = f"{path_prefix}_misc.yaml"

        # Save network state
        try:
            torch.save(self.network.state_dict(), network_path)
            logger.info(f"Network state saved to {network_path}")
        except Exception as e:
            logger.error(f"Error saving network state to {network_path}: {e}")
            return # Stop if network saving fails

        # Save optimizer state
        try:
            torch.save(self.optimizer.state_dict(), optimizer_path)
            logger.info(f"Optimizer state saved to {optimizer_path}")
        except Exception as e:
            logger.error(f"Error saving optimizer state to {optimizer_path}: {e}")
            # Optionally decide if failure here prevents saving misc data

        # Save miscellaneous state (total_steps, config)
        misc_data = {
            "total_steps": self.total_steps,
            "config": self.config,
        }
        try:
            with open(misc_path, 'w') as f:
                yaml.dump(misc_data, f, default_flow_style=False)
            logger.info(f"Miscellaneous state saved to {misc_path} (Steps: {self.total_steps})")
        except Exception as e:
            logger.error(f"Error saving miscellaneous state to {misc_path}: {e}")

    def load_model(self, path_prefix):
        """Loads the agent's state from specified files (network, optimizer, misc)."""
        # This method loads models saved independently (e.g., final models),
        # not checkpoints used for resuming training.
        network_path = f"{path_prefix}_network.pth"
        optimizer_path = f"{path_prefix}_optimizer.pth"
        misc_path = f"{path_prefix}_misc.yaml"

        # Track success of each component separately
        network_loaded = False
        optimizer_loaded = False
        misc_loaded = False

        # Load network state
        if os.path.exists(network_path):
            logger.info(f"Loading network state from: {network_path}")
            try:
                # Load to CPU first to handle device mismatches
                state_dict = torch.load(network_path, map_location='cpu')
                self.network.load_state_dict(state_dict)
                self.network.to(self.device) # Move to the correct device
                # Also load into target network initially
                self.target_network.load_state_dict(self.network.state_dict())
                self.target_network.to(self.device)
                self.target_network.eval()
                logger.info("Network state loaded.")
                network_loaded = True
            except Exception as e:
                logger.error(f"Error loading network state: {e}")
        else:
            logger.warning(f"Network state file not found: {network_path}")

        # Load optimizer state (regardless of network loading success)
        if os.path.exists(optimizer_path):
            logger.info(f"Loading optimizer state from: {optimizer_path}")
            try:
                # Load to CPU first
                state_dict = torch.load(optimizer_path, map_location='cpu')
                self.optimizer.load_state_dict(state_dict)
                # Move optimizer states to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                logger.info("Optimizer state loaded.")
                optimizer_loaded = True
            except Exception as e:
                logger.error(f"Error loading optimizer state: {e}")
        else:
            logger.warning(f"Optimizer state file not found: {optimizer_path}")

        # Load miscellaneous state (total_steps, potentially config)
        config_mismatch = False
        if os.path.exists(misc_path):
            logger.info(f"Loading miscellaneous state from: {misc_path}")
            try:
                with open(misc_path, 'r') as f:
                    misc_data = yaml.safe_load(f)
                
                # Load total_steps for standalone model loading
                if 'total_steps' in misc_data:
                    self.total_steps = misc_data['total_steps']
                    logger.info(f"Miscellaneous data loaded (total_steps applied: {self.total_steps}).")
                    misc_loaded = True
                else:
                    logger.warning("Miscellaneous data loaded but 'total_steps' not found.")

                # Check for config compatibility
                if 'config' in misc_data:
                    if misc_data['config'] != self.config:
                        config_mismatch = True
                        logger.warning("Configuration mismatch detected: Loaded model config differs from current agent config.")
                else:
                    logger.warning("Miscellaneous data loaded but 'config' key not found for comparison.")
            except Exception as e:
                logger.error(f"Error loading miscellaneous state: {e}")
        else:
            logger.warning(f"Miscellaneous state file not found: {misc_path}")

        loaded_successfully = network_loaded and optimizer_loaded and misc_loaded and not config_mismatch
        if loaded_successfully:
            logger.info(f"Model loaded successfully from prefix {path_prefix}")
        else:
            logger.warning(f"Model loading from prefix {path_prefix} encountered issues.")
        
        return loaded_successfully

    # --- NEW METHOD for loading state from unified checkpoint ---
    def load_state(self, agent_state_dict: dict):
        """Loads the agent's state from a dictionary (typically from a unified checkpoint)."""
        logger.info("Loading agent state from dictionary...")
        loaded_successfully = True
        try:
            # Load Network State
            if 'network_state_dict' in agent_state_dict:
                self.network.load_state_dict(agent_state_dict['network_state_dict'])
                self.network.to(self.device)
                logger.info("Network state loaded from dict.")
            else:
                logger.warning("Agent state dict missing 'network_state_dict'.")
                loaded_successfully = False

            # Load Target Network State
            if 'target_network_state_dict' in agent_state_dict:
                self.target_network.load_state_dict(agent_state_dict['target_network_state_dict'])
                self.target_network.to(self.device)
                self.target_network.eval()
                logger.info("Target network state loaded from dict.")
            else:
                logger.warning("Agent state dict missing 'target_network_state_dict'.")
                loaded_successfully = False

            # Load Optimizer State
            if 'optimizer_state_dict' in agent_state_dict:
                self.optimizer.load_state_dict(agent_state_dict['optimizer_state_dict'])
                # Move optimizer states to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                logger.info("Optimizer state loaded from dict.")
            else:
                logger.warning("Agent state dict missing 'optimizer_state_dict'.")
                loaded_successfully = False

            # Load Total Steps
            if 'agent_total_steps' in agent_state_dict:
                self.total_steps = agent_state_dict['agent_total_steps']
                logger.info(f"Agent total_steps loaded from dict: {self.total_steps}")
            else:
                logger.warning("Agent state dict missing 'agent_total_steps'.")
                # Decide if this is critical - maybe allow loading without it?
                # loaded_successfully = False

            # Optional: Could also load/verify config here if needed
            if 'agent_config' in agent_state_dict:
                if agent_state_dict['agent_config'] != self.config:
                    logger.warning("Config from loaded agent state dict differs from current agent config.")
                else:
                    logger.info("Agent config from dict matches current config.")
            else:
                logger.warning("Agent state dict missing 'agent_config'.")
                # loaded_successfully = False # Optional: make config matching mandatory

        except Exception as e:
            logger.error(f"Error loading agent state from dictionary: {e}", exc_info=True)
            loaded_successfully = False

        if loaded_successfully:
            logger.info("Agent state successfully loaded from dictionary.")
        else:
            logger.error("Failed to load agent state completely from dictionary.")

        return loaded_successfully
    # --- END NEW METHOD ---

    def set_training_mode(self, training=True):
        """Sets the agent and network to training or evaluation mode."""
        self.training_mode = training
        mode = "TRAINING" if training else "EVALUATION"
        logger.info(f"Set agent to {mode} mode")
        if self.network:
            if training:
                self.network.train()
            else:
                self.network.eval()
                # Ensure target is also in eval mode (should be already, but safe)
                self.target_network.eval()
