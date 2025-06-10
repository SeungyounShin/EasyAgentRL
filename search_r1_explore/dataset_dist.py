from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import pandas as pd
import tempfile
from huggingface_hub import hf_hub_download

# Define function to align metadata structure
def align_metadata(df):
    # Create standardized metadata structure
    df['metadata'] = df['metadata'].apply(lambda x: {
        "context": {
            "sentences": x.get("context", {}).get("sentences", []),
            "title": x.get("context", {}).get("title", [])
        },
        "level": x.get("level", ""),
        "supporting_facts": {
            "sent_id": x.get("supporting_facts", {}).get("sent_id", []),
            "title": x.get("supporting_facts", {}).get("title", [])
        },
        "type": x.get("type", "")
    })
    return df

# Process train data
with tempfile.TemporaryDirectory() as tmp_download_dir:
    local_parquet_filepath = hf_hub_download(
        repo_id='PeterJinGo/nq_hotpotqa_train',
        filename='train.parquet',
        repo_type="dataset",
        local_dir=tmp_download_dir
    )
    df_raw = pd.read_parquet(local_parquet_filepath)
    train_hotpotqa_raw = df_raw[df_raw['data_source'] == 'hotpotqa']

# Process test data
with tempfile.TemporaryDirectory() as tmp_download_dir:
    local_parquet_filepath = hf_hub_download(
        repo_id='PeterJinGo/nq_hotpotqa_train',
        filename='test.parquet',
        repo_type="dataset",
        local_dir=tmp_download_dir
    )
    df_raw = pd.read_parquet(local_parquet_filepath)
    test_hotpotqa_raw = df_raw[df_raw['data_source'] == 'hotpotqa']

# Align metadata for both datasets
train_hotpotqa_raw = align_metadata(train_hotpotqa_raw)
test_hotpotqa_raw = align_metadata(test_hotpotqa_raw)

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_hotpotqa_raw.reset_index(drop=True))
# filter train_dataset[0]['metadata']['level'] == 'hard'
train_dataset = train_dataset.filter(lambda x: x['metadata']['level'] == 'hard')
test_dataset = Dataset.from_pandas(test_hotpotqa_raw.reset_index(drop=True))

# Create dataset dictionary
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Verify feature alignment
print("Train features:", train_dataset.features)
print("Test features:", test_dataset.features)

# Push to Hub (replace with your repo ID)
dataset_dict.push_to_hub("Seungyoun/search_r1_hotpotqa_train_hard")