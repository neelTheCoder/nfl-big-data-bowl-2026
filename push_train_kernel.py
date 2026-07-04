"""Push the GPU training kernel (neel999/nfl-bdb26-train) via bearer REST."""
import json
import os
from pathlib import Path

import requests

TOKEN = Path(os.path.expanduser("~/.kaggle/access_token")).read_text().strip()
H = {"Authorization": f"Bearer {TOKEN}"}
SRC = Path(__file__).parent / "kaggle_pkg/train_kernel/nfl_bdb26_train.py"

body = {
    "slug": "neel999/nfl-bdb26-train-5-fold-1st-place-scale",
    "newTitle": "NFL BDB26 Train (5-fold 1st-place scale)",
    "text": SRC.read_text(),
    "language": "python",
    "kernelType": "script",
    "isPrivate": True,
    "enableGpu": True,
    # P100 (sm_60) has no kernels in Kaggle's modern torch build -> request T4
    "machineShape": "NvidiaTeslaT4",
    "enableTpu": False,
    "enableInternet": False,
    "datasetDataSources": ["neel999/nfl-bdb26-models"],
    "competitionDataSources": ["nfl-big-data-bowl-2026-prediction"],
    "kernelDataSources": [],
    "modelDataSources": [],
    "categoryIds": [],
}
r = requests.post("https://www.kaggle.com/api/v1/kernels/push", headers=H,
                  json=body, timeout=120)
print(r.status_code, r.text[:800])
