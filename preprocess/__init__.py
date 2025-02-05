from pathlib import Path

COQA_PATH = Path(__file__).parent.parent / 'data' / 'coqa'
QUAC_PATH = Path(__file__).parent.parent / 'data' / 'quac'

COQA_TRAIN_PATH = COQA_PATH / 'coqa-train-v1.0.json'
COQA_DEV_PATH = COQA_PATH / 'coqa-dev-v1.0.json'

QUAC_TRAIN_PATH = QUAC_PATH / 'train_v0.2.json'
QUAC_DEV_PATH = QUAC_PATH / 'val_v0.2.json'