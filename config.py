"""Combat Mission AI Configuration"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class GameWindow:
    """Screen region where the game runs (windowed mode recommended)."""
    left: int = 0
    top: int = 0
    width: int = 1920
    height: int = 1080


@dataclass
class GridConfig:
    """Divide the playable area into a grid for discrete actions."""
    rows: int = 10
    cols: int = 10
    # Exclude UI borders from click area
    margin_top: int = 50
    margin_bottom: int = 100
    margin_left: int = 10
    margin_right: int = 10

@dataclass
class Hotkeys:
    """Combat Mission keyboard bindings."""
    end_turn: str = "enter"
    cycle_unit: str = "tab"
    move_fast: str = "f"
    move_quick: str = "q"
    target: str = "t"
    pause: str = "space"
    camera_top_down: str = "9"  # View 9

@dataclass
class Config:
    window: GameWindow = field(default_factory=GameWindow)
    grid: GridConfig = field(default_factory=GridConfig)
    hotkeys: Hotkeys = field(default_factory=Hotkeys)
    replay_wait_seconds: int = 65  # WeGo turn is 60s + buffer

CONFIG = Config()
