"""Build the four trimmed CSVs in data/ from the original Kaggle download.

Source: "Startup Investments" (Crunchbase 2013 Snapshot), kaggle.com/datasets/justinas/startup-investments
Download the dataset, unzip it into a folder, and run:

    python data/make_extracts.py /path/to/unzipped/kaggle/folder

Why trim: objects.csv is 285 MB (over GitHub's 100 MB limit) and mixes companies with
people, investors and products. This script keeps only company records and only the
columns the notebook uses, and drops every column that names or links to a person
or a company (names, URLs, Twitter handles, descriptions). No rows inside the
company/funding/exit tables are filtered here; all cleaning happens in model.ipynb.
"""
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
out = Path(__file__).resolve().parent

objects = pd.read_csv(src / "objects.csv", low_memory=False)
print("objects.csv rows:", len(objects))
print(objects["entity_type"].value_counts().to_string())

companies = objects.loc[
    objects["entity_type"] == "Company",
    ["id", "category_code", "status", "founded_at", "country_code", "state_code",
     "funding_rounds", "funding_total_usd"],
]
companies.to_csv(out / "companies.csv", index=False)

rounds = pd.read_csv(src / "funding_rounds.csv")
rounds[["funding_round_id", "object_id", "funded_at", "funding_round_type",
        "funding_round_code", "raised_amount_usd", "participants",
        "created_at"]].to_csv(out / "funding_rounds.csv", index=False)

acq = pd.read_csv(src / "acquisitions.csv")
acq[["acquisition_id", "acquired_object_id", "acquired_at"]].to_csv(
    out / "acquisitions.csv", index=False)

ipos = pd.read_csv(src / "ipos.csv")
ipos[["ipo_id", "object_id", "public_at"]].to_csv(out / "ipos.csv", index=False)

for f in ["companies.csv", "funding_rounds.csv", "acquisitions.csv", "ipos.csv"]:
    p = out / f
    print(f, len(pd.read_csv(p, low_memory=False)), "rows,", round(p.stat().st_size / 1e6, 1), "MB")
