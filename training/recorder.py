"""Record human gameplay for imitation learning."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not installed. Run: pip install pynput")

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from config import CONFIG


class GameplayRecorder:
    """
    Records screenshots and mouse/keyboard inputs during human gameplay.
    
    Use this to collect training data for behavior cloning (imitation learning).
    The AI learns to predict your actions given a screenshot.
    
    Usage:
        recorder = GameplayRecorder()
        recorder.start()
        # Play the game...
        # Press Ctrl+C to stop and save
    
    Output:
        data/<timestamp>/
            frame_00001.npy  # Screenshots as numpy arrays
            frame_00002.npy
            ...
            actions.json     # Log of all actions with timestamps
    """
    
    def __init__(self, output_dir: str = "data", capture_on_click: bool = True):
        """
        Initialize the recorder.
        
        Args:
            output_dir: Base directory for recordings
            capture_on_click: If True, capture screenshot on every click.
                             If False, capture on keyboard presses too.
        """
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput is required. Install with: pip install pynput")
        
        # Lazy import to avoid requiring dxcam just to import this module
        from interface.screen import ScreenCapture
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.screen = ScreenCapture()
        self.records = []
        self.frame_count = 0
        self.running = False
        self.capture_on_click = capture_on_click
        
        # Track modifier keys
        self.shift_held = False
        self.ctrl_held = False
        self.alt_held = False
    
    def _capture_frame(self) -> str:
        """Capture and save current screen, return filename."""
        frame = self.screen.grab()
        if frame is not None:
            filename = f"frame_{self.frame_count:05d}.npy"
            np.save(str(self.output_dir / filename), frame)
            self.frame_count += 1
            return filename
        return ""
    
    def _is_in_game_window(self, x: int, y: int) -> bool:
        """Check if coordinates are within the game window."""
        w = CONFIG.window
        return (w.left <= x < w.left + w.width and
                w.top <= y < w.top + w.height)
    
    def _on_click(self, x, y, button, pressed):
        """Mouse click handler."""
        if not pressed or not self.running:
            return
        
        if not self._is_in_game_window(x, y):
            return  # Ignore clicks outside game
        
        # Capture frame before action
        frame_file = self._capture_frame()
        
        # Convert to grid cell (approximate)
        grid = CONFIG.grid
        rel_x = x - CONFIG.window.left - grid.margin_left
        rel_y = y - CONFIG.window.top - grid.margin_top
        
        cell_w = (CONFIG.window.width - grid.margin_left - grid.margin_right) // grid.cols
        cell_h = (CONFIG.window.height - grid.margin_top - grid.margin_bottom) // grid.rows
        
        col = max(0, min(grid.cols - 1, rel_x // cell_w))
        row = max(0, min(grid.rows - 1, rel_y // cell_h))
        cell_id = row * grid.cols + col
        
        self.records.append({
            "frame": frame_file,
            "frame_num": self.frame_count - 1,
            "action_type": "click",
            "x": x,
            "y": y,
            "grid_cell": cell_id,
            "grid_row": row,
            "grid_col": col,
            "button": str(button),
            "modifiers": {
                "shift": self.shift_held,
                "ctrl": self.ctrl_held,
                "alt": self.alt_held,
            },
            "timestamp": time.time()
        })
        
        print(f"[{self.frame_count}] Click at ({x}, {y}) -> grid cell {cell_id}")
    
    def _on_key_press(self, key):
        """Keyboard press handler."""
        if not self.running:
            return
        
        # Track modifier keys
        try:
            if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.shift_held = True
                return
            if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_held = True
                return
            if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_held = True
                return
        except AttributeError:
            pass
        
        # Get key name
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
        
        # Optionally capture frame on keypress
        frame_file = ""
        if not self.capture_on_click:
            frame_file = self._capture_frame()
        
        self.records.append({
            "frame": frame_file,
            "frame_num": self.frame_count - 1 if frame_file else None,
            "action_type": "keypress",
            "key": key_name,
            "modifiers": {
                "shift": self.shift_held,
                "ctrl": self.ctrl_held,
                "alt": self.alt_held,
            },
            "timestamp": time.time()
        })
        
        print(f"[{self.frame_count}] Key: {key_name}")
    
    def _on_key_release(self, key):
        """Track modifier key releases."""
        try:
            if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.shift_held = False
            if key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_held = False
            if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_held = False
        except AttributeError:
            pass
    
    def start(self):
        """Start recording gameplay. Press Ctrl+C to stop."""
        print("=" * 60)
        print("GAMEPLAY RECORDER")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Game window: {CONFIG.window.left},{CONFIG.window.top} "
              f"{CONFIG.window.width}x{CONFIG.window.height}")
        print()
        print("Recording started. Play Combat Mission normally.")
        print("Press Ctrl+C in this terminal to stop and save.")
        print("=" * 60)
        
        self.running = True
        
        mouse_listener = mouse.Listener(on_click=self._on_click)
        keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        mouse_listener.start()
        keyboard_listener.start()
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            mouse_listener.stop()
            keyboard_listener.stop()
            
            # Save action log
            log_path = self.output_dir / "actions.json"
            with open(log_path, "w") as f:
                json.dump(self.records, f, indent=2)
            
            print()
            print("=" * 60)
            print(f"Recording stopped.")
            print(f"Saved {self.frame_count} frames and {len(self.records)} actions")
            print(f"Output: {self.output_dir}")
            print("=" * 60)


if __name__ == "__main__":
    if PYNPUT_AVAILABLE:
        recorder = GameplayRecorder()
        recorder.start()
    else:
        print("Install pynput first: pip install pynput")
