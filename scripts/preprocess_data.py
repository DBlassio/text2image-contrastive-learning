import os
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

data_root = "data/raw"
processed_path = "data/processed/flickr30k_tokens.pkl"

# We download the Flickr30k dataset using Hugging Face.
dataset = load_dataset("lmms-lab/flickr30k")
dataset = dataset["train"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_data = {}
for idx, (img, captions) in enumerate(tqdm(dataset)):
    tokenized_captions = [tokenizer(cap, padding='max_length', truncation=True, max_length=32, return_tensors="pt") for cap in captions]
    tokenized_data[idx] = tokenized_captions

with open(processed_path, "wb") as f:
    pickle.dump(tokenized_data, f)

print("Preprocessed completed and saved to", processed_path)