"""Cross-platform screen capture using mss.

Works on Windows, Linux, and macOS.
Replaces dxcam (Windows-only) for Linux compatibility.
"""
import numpy as np

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not installed. Run: pip install mss")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Run: pip install opencv-python")

import sys
import os

# Handle path for both Windows and Linux
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from config import CONFIG


class ScreenCapture:
    """Cross-platform screen capture using mss.
    
    Works on Linux (X11/Wayland), Windows, and macOS.
    """
    
    def __init__(self):
        if not MSS_AVAILABLE:
            raise RuntimeError("mss is required. Install with: pip install mss")
        
        self.sct = mss.mss()
        self._region = {
            "left": CONFIG.window.left,
            "top": CONFIG.window.top,
            "width": CONFIG.window.width,
            "height": CONFIG.window.height,
        }
    
    def grab(self) -> np.ndarray:
        """
        Capture current frame.
        
        Returns:
            np.ndarray: HxWxC BGR array, or None if capture failed.
        """
        try:
            screenshot = self.sct.grab(self._region)
            # mss returns BGRA, convert to BGR
            frame = np.array(screenshot)[:, :, :3]
            return frame
        except Exception as e:
            print(f"Screen capture failed: {e}")
            return None
    
    def grab_grayscale(self) -> np.ndarray:
        """
        Capture frame as grayscale for observation space.
        
        Returns:
            np.ndarray: HxWx1 grayscale array.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")
        
        frame = self.grab()
        if frame is None:
            # Return black screen if capture failed
            return np.zeros((CONFIG.window.height, CONFIG.window.width, 1), dtype=np.uint8)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray[..., np.newaxis]
    
    def grab_resized(self, width: int = 160, height: int = 90) -> np.ndarray:
        """
        Capture and resize frame for faster processing.
        
        Args:
            width: Target width (default 160 for 8x downscale from 1280)
            height: Target height (default 90 for 8x downscale from 720)
        
        Returns:
            np.ndarray: Resized grayscale array.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv-python is required")
        
        gray = self.grab_grayscale()
        resized = cv2.resize(gray, (width, height))
        return resized[..., np.newaxis]


if __name__ == "__main__":
    # Quick test
    if MSS_AVAILABLE:
        cap = ScreenCapture()
        frame = cap.grab()
        if frame is not None:
            print(f"Captured frame shape: {frame.shape}")
        else:
            print("Failed to capture frame")
    else:
        print("Install mss first: pip install mss")
