import torch

from modules.detector.train import load_detector


def test_connection():
    try:
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        detector = load_detector(checkpoint="best", device=device)

        if detector is None:
            return False, "Detector failed to load"

        return True, f"Detector loaded successfully on {device}"

    except Exception as e:
        return False, str(e)