#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural ranking homework solution"""

import os
import argparse
from timeit import default_timer as timer
import random
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import gc


# Define dataset and model classes (from the provided script)
class VKMarcoDataset(Dataset):
    def __init__(self, data, doc_path, doc_offsets, neg_sampling=1.0):
        if neg_sampling < 1.0:
            high_rel = data[data['Label'] >= 2]
            medium_rel = data[data['Label'] == 1]
            low_rel = data[data['Label'] == 0]

            low_rel = low_rel.sample(frac=neg_sampling, random_state=42)

            self.data = pd.concat([high_rel, medium_rel, low_rel])
        else:
            self.data = data

        self.doc_path = doc_path
        self.doc_offsets = doc_offsets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_text, doc_id, label = self.data.iloc[idx, [1, 2, 3]]
        doc = self.extract_line(int(doc_id[1:]))
        doc_text = f"{doc[2]} {doc[3]}"
        return (query_text.lower(), doc_text.lower()), label / 3.0

    def extract_line(self, index):
        with open(self.doc_path, 'rb') as file:
            file.seek(self.doc_offsets[index - 1])
            line = file.readline().decode('utf-8').strip()
        return line.split('\t')


class RankingModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', layers_to_unfreeze=2):
        super(RankingModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.head = nn.Linear(hidden_size, 1)

        for name, par in self.transformer.named_parameters():
            if 'bias' in name or 'LayerNorm' in name:
                continue
            par.requires_grad = False

        layer_count = self.transformer.config.num_hidden_layers
        for i in range(layers_to_unfreeze):
            for par in self.transformer.encoder.layer[layer_count - 1 - i].parameters():
                par.requires_grad = True

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.transformer(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask
                             )[0][:, 0, :]
        x = self.head(x).squeeze(-1)
        return x


# Utility functions
def compose_batch(batch):
    texts = [x for x, _ in batch]
    ys = torch.tensor([y for _, y in batch]).float()
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    return tokens, ys


def move_batch_to_device(batch, device):
    batch_x, y = batch
    for key in batch_x:
        batch_x[key] = batch_x[key].to(device)
    y = y.to(device)
    return batch_x, y


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Neural ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Measure script execution time
    start = timer()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    TRAIN_QUERIES_PATH = os.path.join(args.data_dir, "vkmarco-doctrain-queries.tsv")
    TRAIN_QRELS_PATH = os.path.join(args.data_dir, "vkmarco-doctrain-qrels.tsv")
    VAL_QUERIES_PATH = os.path.join(args.data_dir, "vkmarco-docdev-queries.tsv")
    VAL_QRELS_PATH = os.path.join(args.data_dir, "vkmarco-docdev-qrels.tsv")
    TEST_QUERIES_PATH = os.path.join(args.data_dir, "/vkmarco-doceval-queries.tsv")
    DOCS_PATH = os.path.join(args.data_dir, "vkmarco-docs.tsv")

    # Load and preprocess data
    df_queries_train = pd.read_csv(TRAIN_QUERIES_PATH, sep='\t', header=None, names=['QueryId', 'Query'])
    df_queries_val = pd.read_csv(VAL_QUERIES_PATH, sep='\t', header=None, names=['QueryId', 'Query'])
    df_queries_test = pd.read_csv(TEST_QUERIES_PATH, sep='\t', header=None, names=['QueryId', 'Query'])
    df_qrels_train = pd.read_csv(TRAIN_QRELS_PATH, sep=' ', header=None,
                                 names=['QueryId', 'unused', 'DocumentId', 'Label'])
    df_qrels_val = pd.read_csv(VAL_QRELS_PATH, sep=' ', header=None, names=['QueryId', 'unused', 'DocumentId', 'Label'])

    train_data = pd.merge(df_queries_train, df_qrels_train, how="right", on="QueryId").drop(columns="unused")
    val_data = pd.merge(df_queries_val, df_qrels_val, how="right", on="QueryId").drop(columns="unused")

    doc_offsets = []
    with open(DOCS_PATH, 'rb') as file:
        position = 0
        for line in file:
            doc_offsets.append(position)
            position = file.tell()

    dataset_train = VKMarcoDataset(train_data, DOCS_PATH, doc_offsets, neg_sampling=0.3)
    dataset_valid = VKMarcoDataset(val_data, DOCS_PATH, doc_offsets)
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=256, collate_fn=compose_batch)
    val_loader = DataLoader(dataset_valid, shuffle=False, batch_size=256, collate_fn=compose_batch)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = RankingModel('xlm-roberta-base')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_steps = 0

        progress_bar = tqdm(
            enumerate(train_loader),
            desc=f'Epoch {epoch + 1}/{num_epochs}',
            total=len(train_loader.dataset) // train_loader.batch_size,
            unit=' batch',
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )

        for step, batch in progress_bar:
            tokenized, labels = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(**tokenized)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            for param in model.transformer.parameters():
                if not param.requires_grad:
                    param.grad = None

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_steps += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / train_steps:.4f}',
                'batch': f'{step + 1}'
            })

            if step % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        avg_train_loss = total_loss / train_steps

        print(f'\nEpoch {epoch + 1} Results:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')

    sample_submission_path = os.path.join(args.data_dir, 'sample_submission.csv')
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission['Label'] = 0

    test_data = pd.merge(df_queries_test, sample_submission, how="right", on="QueryId")
    dataset_test = VKMarcoDataset(test_data, DOCS_PATH, doc_offsets)
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=512, collate_fn=compose_batch)

    model.eval()

    y_test = []
    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader.dataset) // test_loader.batch_size,
        unit='batch',
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
    )

    for i, batch in progress_bar:
        tokenized, _ = move_batch_to_device(batch, device)
        with torch.no_grad():
            preds = model(**tokenized)
            y_test.extend(preds)

    numpy_array = torch.stack(y_test).cpu().tolist()
    test_data['pred'] = numpy_array
    result_df = test_data.sort_values(by=['QueryId', 'pred'], ascending=[True, False])

    result_df[['QueryId', 'DocumentId']].to_csv('submission.csv', index=False)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}s")


if __name__ == "__main__":
    main()
