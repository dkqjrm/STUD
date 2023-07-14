from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import re


class StudDataset(Dataset):
    def __init__(self, path, prompt, length=1, oversample=0):
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.length = length
        self.prompt = prompt
        self.oversample = oversample

        # Load data and conditionally perform oversampling
        df = pd.read_csv(path, encoding='utf-8-sig', sep='\t')
        df = df.dropna()
        if 'train' in path:
            self.oversample_data(df)
        else:
            self.preprocessing(df)

        # Print class distribution after oversampling (if performed)
        class_counts = np.bincount(self.label_data)
        print("Class counts after oversampling:", class_counts)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        # print(text)
        label = self.label_data[idx]
        input_data = self.tokenizer.encode_plus(self.prompt, text,
                                                return_tensors='pt',
                                                padding='max_length',
                                                max_length=512-self.length,
                                                truncation=True)
        # print(self.tokenizer.decode(input_data['input_ids'][0]))
        input_data.update({"label": torch.tensor(label)})
        input_data = {k: v.squeeze() for k, v in input_data.items()}

        return input_data

    def oversample_data(self, df):
        # Calculate the number of examples in each class
        class_counts = np.bincount(df['label'].astype(int))

        # Identify the minority and majority class
        minority_class = np.argmin(class_counts)
        majority_class = np.argmax(class_counts)

        # Oversample the minority class
        minority_data = df[df['label'] == minority_class]
        oversampled_data = minority_data.sample(n=int(self.oversample*class_counts[majority_class]), replace=True)

        # Combine the oversampled data with the original data
        df_balanced = pd.concat([df, oversampled_data], ignore_index=True)

        # Preprocess the data
        self.text_data = []
        self.label_data = []
        for seq in tqdm(df_balanced.iloc, total=df_balanced.shape[0], desc="preprocessing"):
            tmp = list(seq)
            self.text_data.append(str(tmp[0])+' '+str(tmp[1])+' '+str(tmp[2])+' '+str(tmp[3]))
            self.label_data.append(int(tmp[4]))

    def preprocessing(self, df):
        self.text_data = []
        self.label_data = []

        for seq in tqdm(df.iloc, total=df.shape[0], desc="preprocessing"):
            tmp = list(seq)
            text = str(tmp[0]) + ' ' + str(tmp[1]) + ' ' + str(tmp[2])# + ' ' + str(tmp[3])
            self.text_data.append(text)
            self.label_data.append(int(tmp[4]))

    def custom_collate(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        label = torch.stack([item['label'] for item in batch])

        # Calculate max length of sequences in batch
        max_length = torch.max(torch.sum(attention_mask, dim=1)).item()

        return {'input_ids': input_ids[:, :max_length],
                'token_type_ids' : token_type_ids[:, :max_length],
                'attention_mask': attention_mask[:, :max_length],
                'label': label}

if __name__ == "__main__":
    dataset = StudDataset(path='../val.tsv',prompt='',length=64,oversample=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate)

    for i in dataloader:
        # print(i)
        print({k: v.shape for k, v in i.items()})
        print({k: v for k, v in i.items()})
        break
