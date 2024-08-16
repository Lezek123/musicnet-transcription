import os
import shutil
from pathlib import Path


PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)
IS_CLOUD = bool(os.environ.get("CLOUD_ML_PROJECT_ID"))

def create_vocab(value_set):
    values = list(value_set)
    values.sort()
    return { value: index for index, value in enumerate(values) }

def recreate_dirs(dirs: list[str]):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True, mode=0o775)

# TODO: Adjust
# def note_frequency(note_idx):
#     note = list(notes_vocab.keys())[note_idx]
#     return 440 * (2 ** ((note - 69) / 12))