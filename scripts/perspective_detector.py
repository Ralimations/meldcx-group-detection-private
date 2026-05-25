import numpy as np
from config import OpenVinoDefaults

class PerspectiveDetector:
    """
    Object-Oriented encapsulation for enforcing the explicitly configured camera perspective.
    """
    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.mode = getattr(cfg, "perspective", "LEVELED").upper()
        if self.mode not in ("LEVELED", "TOP-DOWN"):
            print(f"[Perspective] WARNING: Unknown perspective '{self.mode}'. Enforcing 'LEVELED'.")
            self.mode = "LEVELED"

    def update(self, raw_boxes: np.ndarray) -> str:
        """
        Returns the statically configured perspective view.
        (Retained for API compatibility with main pipeline).
        """
        return self.mode
        
    def get_current_view(self) -> str:
        """Returns the active perspective view."""
        return self.mode
