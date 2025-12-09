from pathlib import Path

DATA_PATH = Path('data')
PAD_20_PLUS_PATH = Path("/home/pedrobouzon/life/datasets/pad-ufes-20-plus")
PAD_20_PLUS_IMAGES_FOLDER = PAD_20_PLUS_PATH / "images"
PAD_20_PLUS_ONE_HOT_ENCODED = DATA_PATH/ "one-hot.csv"
PAD_20_PLUS_RAW_METADATA = PAD_20_PLUS_PATH / "metadata.csv"