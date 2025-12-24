"""Gymnasium Environment for Combat Mission."""
import time
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed. Run: pip install gymnasium")
    # Create dummy classes for import
    class gym:
        class Env: pass
    class spaces:
        @staticmethod
        def Discrete(n): return None
        @staticmethod
        def Box(**kwargs): return None

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from config import CONFIG


class CombatMissionEnv(gym.Env):
    """
    Gymnasium Environment wrapping Combat Mission for RL training.
    
    This environment:
    - Captures the game screen as observations
    - Executes discrete actions (grid clicks + hotkeys)
    - Calculates rewards based on game state changes
    
    Action Space (Discrete):
        0 to (grid_size-1): Click grid cell (for move/target orders)
        grid_size: Cycle to next unit (Tab)
        grid_size+1: Move Fast command (F)
        grid_size+2: Move Quick command (Q)  
        grid_size+3: Target command (T)
        grid_size+4: End Turn (Enter) - executes orders and waits for replay
    
    Observation Space:
        Grayscale screen capture resized to 160x90 for efficiency.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, use_small_obs: bool = True):
        """
        Initialize the Combat Mission environment.
        
        Args:
            use_small_obs: If True, observations are 160x90 grayscale.
                          If False, full 1280x720 grayscale.
        """
        super().__init__()
        
        # Lazy imports to allow module loading without deps
        from interface.screen import ScreenCapture
        from interface.input import InputController
        from interface.ocr import OCRReader
        
        self.screen = ScreenCapture()
        self.input = InputController()
        
        try:
            self.ocr = OCRReader()
        except Exception:
            self.ocr = None  # OCR optional
        
        self.grid_size = CONFIG.grid.rows * CONFIG.grid.cols
        self.use_small_obs = use_small_obs
        
        # Action space: grid cells + 5 hotkey actions
        # [0..99]: Grid clicks
        # [100]: Cycle unit
        # [101]: Move Fast
        # [102]: Move Quick
        # [103]: Target
        # [104]: End Turn
        self.action_space = spaces.Discrete(self.grid_size + 5)
        
        # Observation space: grayscale screen
        if use_small_obs:
            obs_shape = (90, 160, 1)  # Downscaled
        else:
            obs_shape = (CONFIG.window.height, CONFIG.window.width, 1)
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        
        # State tracking
        self.prev_score = 0
        self.turn_count = 0
        self.max_turns = 50  # Episode truncation
        self.action_count = 0
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Note: This doesn't actually restart the game - you need to manually
        load a save or start a new scenario. Future versions could automate this.
        """
        super().reset(seed=seed)
        
        self.prev_score = 0
        self.turn_count = 0
        self.action_count = 0
        
        # Set camera to top-down view
        self.input.set_camera_top_down()
        time.sleep(0.2)
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: int):
        """
        Execute an action and return the new state.
        
        Args:
            action: Integer action from action_space
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.action_count += 1
        
        # Execute action
        if action < self.grid_size:
            # Grid click - typically used for issuing movement/targeting orders
            self.input.click_grid(action)
        elif action == self.grid_size:
            self.input.cycle_unit()
        elif action == self.grid_size + 1:
            self.input.move_fast()
        elif action == self.grid_size + 2:
            self.input.move_quick()
        elif action == self.grid_size + 3:
            self.input.target()
        elif action == self.grid_size + 4:
            # End turn - this triggers the replay phase
            self.input.end_turn()
            self._wait_for_replay()
            self.turn_count += 1
        
        # Small delay for game to respond
        time.sleep(0.05)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = False  # TODO: Detect victory/defeat screen
        truncated = self.turn_count >= self.max_turns
        
        info = {
            "turn": self.turn_count,
            "actions": self.action_count,
            "score": self.prev_score,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Capture current screen state."""
        if self.use_small_obs:
            return self.screen.grab_resized(160, 90)
        else:
            return self.screen.grab_grayscale()
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on game state changes.
        
        This is a placeholder - you'll want to customize based on:
        - Score changes (read via OCR)
        - Unit casualties
        - Objective captures
        - etc.
        """
        if self.ocr is None:
            return 0.0
        
        frame = self.screen.grab()
        if frame is None:
            return 0.0
        
        current_score = self.ocr.extract_score(frame)
        reward = float(current_score - self.prev_score)
        self.prev_score = current_score
        
        return reward
    
    def _wait_for_replay(self):
        """
        Wait for the WeGo replay phase to finish.
        
        CM runs a 60-second replay showing order execution.
        This is a simple fixed wait - a smarter version would
        detect when the Command Phase UI reappears.
        """
        time.sleep(CONFIG.replay_wait_seconds)
    
    def render(self):
        """Render is handled by the game itself."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    if GYM_AVAILABLE:
        print("Creating CombatMissionEnv...")
        try:
            env = CombatMissionEnv(use_small_obs=True)
            print(f"Action space: {env.action_space}")
            print(f"Observation space: {env.observation_space}")
            
            obs, info = env.reset()
            print(f"Initial observation shape: {obs.shape}")
        except Exception as e:
            print(f"Failed to create environment: {e}")
            print("Make sure dxcam and pydirectinput are installed.")
    else:
        print("Install gymnasium first: pip install gymnasium")
