import torch
from torch.utils.data import Dataset


class ParaphraseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.sentence1 = df["sentence1"].tolist()
        self.sentence2 = df["sentence2"].tolist()
        self.labels = df["label"].tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentence1[idx],
            self.sentence2[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }