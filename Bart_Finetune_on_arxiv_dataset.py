#Bibek Shiwakoti

# installing the required libraries to get the dataset
# pip install transformers datasets torch

import torch
from datasets import load_dataset

# dataset = load_dataset("ccdv/arxiv-summarization")

train_arxiv_dataset = load_dataset("ccdv/arxiv-summarization", split="train", cache_dir="cache")
eval_arxiv_dataset = load_dataset("ccdv/arxiv-summarization", split="validation", cache_dir="cache")