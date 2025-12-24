"""OCR utilities for reading in-game text."""
import numpy as np

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Run: pip install pytesseract")


class OCRReader:
    """
    Optical Character Recognition for reading Combat Mission UI text.
    
    Requires Tesseract OCR binary to be installed:
    https://github.com/tesseract-ocr/tesseract
    """
    
    def __init__(self, tesseract_path: str = None):
        if not PYTESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract is required. Install with: pip install pytesseract")
        
        # Auto-detect tesseract path based on OS
        if tesseract_path is None:
            import platform
            if platform.system() == "Windows":
                tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            else:
                # Linux/Mac - tesseract should be in PATH
                tesseract_path = "tesseract"
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self._tesseract_path = tesseract_path

    
    def read_region(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """
        Extract text from a specific region of the screen.
        
        Args:
            frame: BGR or grayscale numpy array (from ScreenCapture)
            x, y: Top-left corner of region
            w, h: Width and height of region
        
        Returns:
            Extracted text string, stripped of whitespace.
        """
        region = frame[y:y+h, x:x+w]
        img = Image.fromarray(region)
        # PSM 7: Treat the image as a single text line
        text = pytesseract.image_to_string(img, config='--psm 7')
        return text.strip()
    
    def read_region_digits(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """
        Extract only digits from a region (for scores, ammo counts, etc.)
        
        Args:
            frame: BGR or grayscale numpy array
            x, y: Top-left corner of region
            w, h: Width and height of region
        
        Returns:
            String containing only digits.
        """
        region = frame[y:y+h, x:x+w]
        img = Image.fromarray(region)
        # Whitelist only digits
        text = pytesseract.image_to_string(
            img, 
            config='--psm 7 -c tessedit_char_whitelist=0123456789'
        )
        return text.strip()
    
    def extract_score(self, frame: np.ndarray, 
                      x: int = 1100, y: int = 10, w: int = 150, h: int = 40) -> int:
        """
        Extract the score from the UI.
        
        Note: Region coordinates need to be calibrated for your CM resolution.
        Default values are estimates for 1280x720.
        
        Args:
            frame: Screen capture
            x, y, w, h: Score region (customize per game)
        
        Returns:
            Score as integer, or 0 if extraction failed.
        """
        text = self.read_region_digits(frame, x, y, w, h)
        try:
            return int(text) if text else 0
        except ValueError:
            return 0


if __name__ == "__main__":
    if PYTESSERACT_AVAILABLE:
        print(f"pytesseract available. Path: {pytesseract.pytesseract.tesseract_cmd}")
        # Try to verify tesseract binary
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except Exception as e:
            print(f"Tesseract binary not found: {e}")
            print("Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    else:
        print("Install pytesseract first: pip install pytesseract")
