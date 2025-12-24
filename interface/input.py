"""Cross-platform input injection using pynput.

Works on Windows, Linux, and macOS.
Replaces pydirectinput (Windows-only) for Linux compatibility.
"""
import time

try:
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not installed. Run: pip install pynput")

import sys
import os

# Handle path for both Windows and Linux
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from config import CONFIG


class InputController:
    """
    Cross-platform mouse/keyboard control for Combat Mission.
    
    Uses pynput which works on Linux (X11), Windows, and macOS.
    Note: On Linux, this requires X11 (not Wayland) or root for /dev/uinput.
    """
    
    def __init__(self):
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput is required. Install with: pip install pynput")
        
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        
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
        self.click_screen(x, y, button)
    
    def click_screen(self, x: int, y: int, button: str = "left"):
        """Click at absolute screen coordinates."""
        self.mouse.position = (x, y)
        time.sleep(0.01)  # Small delay for position to register
        btn = Button.left if button == "left" else Button.right
        self.mouse.click(btn)
    
    def press_key(self, key: str):
        """Press and release a keyboard key."""
        # Handle special keys
        special_keys = {
            'tab': Key.tab,
            'enter': Key.enter,
            'return': Key.enter,
            'space': Key.space,
            'escape': Key.esc,
            'esc': Key.esc,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
        }
        
        if key.lower() in special_keys:
            self.keyboard.press(special_keys[key.lower()])
            self.keyboard.release(special_keys[key.lower()])
        else:
            self.keyboard.press(key)
            self.keyboard.release(key)
    
    def hold_key(self, key: str, duration: float = 0.1):
        """Hold a key for a duration."""
        special_keys = {
            'tab': Key.tab,
            'enter': Key.enter,
            'space': Key.space,
            'escape': Key.esc,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
        }
        
        k = special_keys.get(key.lower(), key)
        self.keyboard.press(k)
        time.sleep(duration)
        self.keyboard.release(k)
    
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
    # Quick test
    if PYNPUT_AVAILABLE:
        print("Test: Will type 'a' in 3 seconds. Open a text editor...")
        time.sleep(3)
        controller = InputController()
        controller.press_key('a')
        print("Done. Did 'a' appear?")
    else:
        print("Install pynput first: pip install pynput")
