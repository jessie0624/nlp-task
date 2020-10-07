from pathlib import Path
from posix import EX_SOFTWARE

PROJECT_DIR = Path(__file__).parent.parent.parent
ROOT_DIR = Path("~/.nlp").expanduser().resolve()
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
CONFIG_DIR = PROJECT_DIR / "configs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class Const:
    example_id = "example_id"

    text = "text"

    tokens = "tokens"
    tokens_id = "tokens_id"

    word_pieces = "word_pieces"
    attn_mask = "attn_mask"
    types_id = "types_id"

    label = "label"
    label_id = "label_id"

    seq_len = "seq_len"

    pred = "pred"
    loss = "loss"
    prob = "prob"     