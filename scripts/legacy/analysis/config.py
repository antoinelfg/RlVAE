"""
Configuration file for Riemannian Flow VAE.
Contains default paths and settings for the clean repository.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PRETRAINED_DIR = DATA_DIR / "pretrained"

# Specific file paths
SPRITES_TRAIN_PATH = RAW_DATA_DIR / "Sprites_train.pt"
SPRITES_TEST_PATH = RAW_DATA_DIR / "Sprites_test.pt"

CYCLIC_TRAIN_PATH = PROCESSED_DATA_DIR / "Sprites_train_cyclic.pt"
CYCLIC_TEST_PATH = PROCESSED_DATA_DIR / "Sprites_test_cyclic.pt"
CYCLIC_TRAIN_META_PATH = PROCESSED_DATA_DIR / "Sprites_train_cyclic_metadata.pt"
CYCLIC_TEST_META_PATH = PROCESSED_DATA_DIR / "Sprites_test_cyclic_metadata.pt"

ENCODER_PATH = PRETRAINED_DIR / "encoder.pt"
DECODER_PATH = PRETRAINED_DIR / "decoder.pt"
METRIC_PATH = PRETRAINED_DIR / "metric.pt"
METRIC_SCALED_PATH = PRETRAINED_DIR / "metric_T0.7_scaled.pt"

# Model defaults
DEFAULT_LATENT_DIM = 16
DEFAULT_INPUT_DIM = (3, 64, 64)
DEFAULT_N_FLOWS = 5
DEFAULT_FLOW_HIDDEN_SIZE = 128

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_BETA = 1.0
DEFAULT_RIEMANNIAN_BETA = 0.5

# Paths validation
def validate_paths():
    """Validate that all required paths exist."""
    required_paths = [
        SPRITES_TRAIN_PATH,
        SPRITES_TEST_PATH,
        CYCLIC_TRAIN_PATH,
        CYCLIC_TEST_PATH,
        ENCODER_PATH,
        DECODER_PATH,
        METRIC_PATH,
        METRIC_SCALED_PATH,
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required files:")
        for path in missing_paths:
            print(f"   {path}")
        return False
    
    print("✅ All required files found")
    return True

if __name__ == "__main__":
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📊 Data directory: {DATA_DIR}")
    print(f"🧠 Source directory: {SRC_DIR}")
    validate_paths() 