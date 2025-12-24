"""Input injection using DirectInput scan codes."""
import time

try:
    import pydirectinput
    pydirectinput.FAILSAFE = False  # Disable corner abort
    PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    PYDIRECTINPUT_AVAILABLE = False
    print("Warning: pydirectinput not installed. Run: pip install pydirectinput")

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from config import CONFIG


class InputController:
    """
    Hardware-level mouse/keyboard control for Combat Mission.
    
    Uses pydirectinput to send DirectInput scan codes,
    which work with DirectX games that ignore virtual keys.
    """
    
    def __init__(self):
        if not PYDIRECTINPUT_AVAILABLE:
            raise RuntimeError("pydirectinput is required. Install with: pip install pydirectinput")
        
        self.grid = CONFIG.grid
        self.window = CONFIG.window
        self.hotkeys = CONFIG.hotkeys
        
        # Pre-calculate cell dimensions
        playable_width = self.window.width - self.grid.margin_left - self.grid.margin_right
        playable_height = self.window.height - self.grid.margin_top - self.grid.margin_bottom
        self._cell_width = playable_width // self.grid.cols
        self._cell_height = playable_height // self.grid.rows
    
    def grid_to_screen(self, row: int, col: int) -> tuple[int, int]:
        """
        Convert grid cell to screen pixel coordinates (center of cell).
        
        Args:
            row: Grid row (0 to rows-1)
            col: Grid column (0 to cols-1)
        
        Returns:
            Tuple of (x, y) screen coordinates.
        """
        x = self.window.left + self.grid.margin_left + col * self._cell_width + self._cell_width // 2
        y = self.window.top + self.grid.margin_top + row * self._cell_height + self._cell_height // 2
        return x, y
    
    def click_grid(self, cell_id: int, button: str = "left"):
        """
        Click on a grid cell by flat index.
        
        Args:
            cell_id: Flat index (0 to rows*cols - 1)
            button: Mouse button ("left" or "right")
        """
        row = cell_id // self.grid.cols
        col = cell_id % self.grid.cols
        x, y = self.grid_to_screen(row, col)
        pydirectinput.click(x, y, button=button)
    
    def click_screen(self, x: int, y: int, button: str = "left"):
        """Click at absolute screen coordinates."""
        pydirectinput.click(x, y, button=button)
    
    def press_key(self, key: str):
        """Press and release a keyboard key."""
        pydirectinput.press(key)
    
    def hold_key(self, key: str, duration: float = 0.1):
        """Hold a key for a duration."""
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
    
    # Combat Mission specific actions
    def end_turn(self):
        """Press the End Turn / Go button."""
        self.press_key(self.hotkeys.end_turn)
    
    def cycle_unit(self):
        """Cycle to next unit."""
        self.press_key(self.hotkeys.cycle_unit)
    
    def move_fast(self):
        """Select Move Fast command."""
        self.press_key(self.hotkeys.move_fast)
    
    def move_quick(self):
        """Select Move Quick command."""
        self.press_key(self.hotkeys.move_quick)
    
    def target(self):
        """Select Target command."""
        self.press_key(self.hotkeys.target)
    
    def set_camera_top_down(self):
        """Set camera to top-down view (View 9)."""
        self.press_key(self.hotkeys.camera_top_down)


if __name__ == "__main__":
    # Quick test: type a letter in an open text editor
    if PYDIRECTINPUT_AVAILABLE:
        print("Test: Will type 'a' in 3 seconds. Open Notepad or similar...")
        time.sleep(3)
        controller = InputController()
        controller.press_key('a')
        print("Done. Did 'a' appear?")
    else:
        print("Install pydirectinput first: pip install pydirectinput")
