from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")

# We must import Kaggle after loading the environment file
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    "msambare/fer2013",
    path="Computer Vision/data/fer2013",
    unzip=True,
)
