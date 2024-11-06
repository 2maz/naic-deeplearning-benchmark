#!/usr/bin/env python

from datasets import load_dataset
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--target-dir", required=True, default="/data/transformer-xl/wikitext-103")

args, unknown = parser.parse_known_args()

base_path = Path(args.target_dir)
base_path.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
for i in ["train", "test", "validation"]:
    dataset = ds[i].data["text"]

    if i == "validation":
        filename = base_path / "valid.txt"
    else:
        filename = base_path / f"{i}.txt"

    with open(filename, 'w') as f:
        for sample in dataset:
            f.write(str(sample) + "\n")

