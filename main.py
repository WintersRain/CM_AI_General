"""Combat Mission AI Agent - Main Entry Point"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Combat Mission AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test         Test screen capture and input
  python main.py record       Record gameplay for training
  python main.py play         Run the trained AI agent
  python main.py train        Train the AI from recorded data
        """
    )
    
    parser.add_argument(
        "command",
        choices=["test", "record", "play", "train"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    if args.command == "test":
        test_environment()
    elif args.command == "record":
        record_gameplay()
    elif args.command == "play":
        run_agent()
    elif args.command == "train":
        train_agent()


def test_environment():
    """Test that all components are working."""
    print("=" * 60)
    print("COMBAT MISSION AI - ENVIRONMENT TEST")
    print("=" * 60)
    
    # Test screen capture
    print("\n[1/3] Testing screen capture...")
    try:
        from interface.screen import ScreenCapture
        cap = ScreenCapture()
        frame = cap.grab()
        if frame is not None:
            print(f"  [OK] Screen capture working: {frame.shape}")
        else:
            print("  [FAIL] Screen capture returned None")
    except Exception as e:
        print(f"  [FAIL] Screen capture failed: {e}")
    
    # Test input controller
    print("\n[2/3] Testing input controller...")
    try:
        from interface.input import InputController
        controller = InputController()
        print(f"  [OK] Input controller initialized")
        print(f"    Grid: {controller.grid.rows}x{controller.grid.cols}")
        x, y = controller.grid_to_screen(5, 5)
        print(f"    Cell (5,5) -> screen ({x}, {y})")
    except Exception as e:
        print(f"  [FAIL] Input controller failed: {e}")
    
    # Test OCR
    print("\n[3/3] Testing OCR...")
    try:
        from interface.ocr import OCRReader
        ocr = OCRReader()
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"  [OK] Tesseract version: {version}")
    except Exception as e:
        print(f"  [FAIL] OCR failed: {e}")
        print("    Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    print("\n" + "=" * 60)
    print("Test complete. Fix any [FAIL] items before proceeding.")
    print("=" * 60)


def record_gameplay():
    """Start recording gameplay for training."""
    from training.recorder import GameplayRecorder
    recorder = GameplayRecorder()
    recorder.start()


def run_agent():
    """Run the trained AI agent."""
    print("AI agent not yet implemented. Train a model first with 'python main.py train'")
    # TODO: Load trained model and run inference loop


def train_agent():
    """Train the AI from recorded data."""
    print("Training not yet implemented. Record gameplay first with 'python main.py record'")
    # TODO: Implement behavior cloning / RL training


if __name__ == "__main__":
    main()
