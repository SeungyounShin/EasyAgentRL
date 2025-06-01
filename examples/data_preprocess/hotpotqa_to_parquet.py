"""Preprocess the HotpotQA dataset to parquet format."""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

SRC_FILE = os.path.join(
    "data",
    "qa_dataset",
    "train",
    "hotpotqa_1000_20250402.json",
)


def load_examples(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("examples", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/hotpotqa")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    examples = load_examples(SRC_FILE)
    random.seed(42)
    random.shuffle(examples)

    split_idx = int(len(examples) * 0.8)
    train_records = []
    test_records = []

    for idx, ex in enumerate(examples):
        dataset_name = ex.get("dataset_name", "hotpotqa/hotpot_qa")
        data_source_tagged = "searchR1_" + dataset_name.split("/")[0]

        row = {
            "data_source": data_source_tagged,
            "prompt": [{"role": "user", "content": ex["question"]}],
            "ability": "nlp",
            "reward_model": {"style": "rule", "ground_truth": ex["answer"]},
            "extra_info": {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "level": ex.get("level", ""),
            },
        }
        if idx < split_idx:
            row["extra_info"]["split"] = "train"
            row["extra_info"]["index"] = idx
            train_records.append(row)
        else:
            row["extra_info"]["split"] = "test"
            row["extra_info"]["index"] = idx - split_idx
            test_records.append(row)

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
