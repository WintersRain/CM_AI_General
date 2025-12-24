"""Cross-platform screen capture.

Uses scrot on Linux (works with Hyper-V) and mss as fallback.
"""
import numpy as np
import subprocess
import tempfile
import os
import time

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: pillow not installed. Run: pip install pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Run: pip install opencv-python")

import sys

# Handle path for both Windows and Linux
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from config import CONFIG


class ScreenCapture:
    """Screen capture using scrot (Linux) with mss fallback.
    
    scrot works reliably on Hyper-V VMs where mss/XGetImage fails.
    """
    
    def __init__(self):
        if not PIL_AVAILABLE:
            raise RuntimeError("pillow is required. Install with: pip install pillow")
        
        self._region = {
            "left": CONFIG.window.left,
            "top": CONFIG.window.top,
            "width": CONFIG.window.width,
            "height": CONFIG.window.height,
        }
        
        # Check if scrot is available
        self._use_scrot = self._check_scrot()
        if self._use_scrot:
            print("Using scrot for screen capture")
        else:
            print("scrot not found, trying mss...")
            try:
                import mss
                self.sct = mss.mss()
            except ImportError:
                raise RuntimeError("Neither scrot nor mss available for screen capture")
    
    def _check_scrot(self) -> bool:
        """Check if scrot is installed."""
        try:
            result = subprocess.run(['which', 'scrot'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def grab(self) -> np.ndarray:
        """
        Capture current frame.
        
        Returns:
            np.ndarray: HxWxC BGR array, or None if capture failed.
        """
        if self._use_scrot:
            return self._grab_scrot()
        else:
            return self._grab_mss()
    
    def _grab_scrot(self) -> np.ndarray:
        """Capture using scrot."""
        tmp_path = f"/tmp/cm_capture_{os.getpid()}.png"
        
        try:
            # Remove old file if exists
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Capture screen with scrot - use overwrite flag
            result = subprocess.run(
                ['scrot', '-o', tmp_path],  # -o = overwrite
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print(f"scrot error: {result.stderr.decode()}")
                return None
            
            # Wait for file to be written
            time.sleep(0.05)
            
            # Verify file exists and has content
            if not os.path.exists(tmp_path):
                print("scrot: output file not created")
                return None
            
            if os.path.getsize(tmp_path) == 0:
                print("scrot: output file is empty")
                return None
            
            # Load and crop to region
            img = Image.open(tmp_path)
            
            # Crop to configured region
            left = self._region["left"]
            top = self._region["top"]
            right = left + self._region["width"]
            bottom = top + self._region["height"]
            
            # Ensure crop bounds are within image
            img_width, img_height = img.size
            right = min(right, img_width)
            bottom = min(bottom, img_height)
            
            if right > left and bottom > top:
                img = img.crop((left, top, right, bottom))
            
            # Convert to numpy BGR
            frame = np.array(img)
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]  # Drop alpha
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if CV2_AVAILABLE:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
            
        except subprocess.TimeoutExpired:
            print("scrot: timeout")
            return None
        except Exception as e:
            print(f"scrot capture failed: {e}")
            return None
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    def _grab_mss(self) -> np.ndarray:
        """Capture using mss (fallback)."""
        try:
            screenshot = self.sct.grab(self._region)
            frame = np.array(screenshot)[:, :, :3]
            return frame
        except Exception as e:
            print(f"mss capture failed: {e}")
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
    cap = ScreenCapture()
    frame = cap.grab()
    if frame is not None:
        print(f"Captured frame shape: {frame.shape}")
        # Save test image
        if CV2_AVAILABLE:
            cv2.imwrite("/tmp/capture_test.png", frame)
            print("Saved test capture to /tmp/capture_test.png")
    else:
        print("Failed to capture frame")
