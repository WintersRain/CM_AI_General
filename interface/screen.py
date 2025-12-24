"""High-performance screen capture via DirectX."""
import numpy as np

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    print("Warning: dxcam not installed. Run: pip install dxcam")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Run: pip install opencv-python")

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])
from config import CONFIG


class ScreenCapture:
    """GPU-accelerated screen capture using dxcam."""
    
    def __init__(self):
        if not DXCAM_AVAILABLE:
            raise RuntimeError("dxcam is required. Install with: pip install dxcam")
        
        self.camera = dxcam.create(output_color="BGR")
        self._region = (
            CONFIG.window.left,
            CONFIG.window.top,
            CONFIG.window.left + CONFIG.window.width,
            CONFIG.window.top + CONFIG.window.height,
        )
    
    def grab(self) -> np.ndarray:
        """
        Capture current frame.
        
        Returns:
            np.ndarray: HxWxC BGR array, or None if capture failed.
        """
        frame = self.camera.grab(region=self._region)
        if frame is None:
            # Retry once
            frame = self.camera.grab(region=self._region)
        return frame
    
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
    if DXCAM_AVAILABLE:
        cap = ScreenCapture()
        frame = cap.grab()
        if frame is not None:
            print(f"Captured frame shape: {frame.shape}")
        else:
            print("Failed to capture frame")
    else:
        print("Install dxcam first: pip install dxcam")
