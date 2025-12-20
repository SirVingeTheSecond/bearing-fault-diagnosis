import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Print utils
# =============================================================================

def print_separator(char: str = "=", width: int = 70):
    """Print a separator line."""
    print(char * width)


def print_header(text: str, char: str = "=", width: int = 70):
    """Print a header with separators above and below."""
    print(char * width)
    print(text)
    print(char * width)


def print_section(text: str, char: str = "-"):
    """Print a section header."""
    print(f"\n{char * 3} {text} {char * 3}")
