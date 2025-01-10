#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural ranking homework solution"""

import argparse
from timeit import default_timer as timer
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import gc
from collections import defaultdict

class VKMarcoIterableDataset(IterableDataset):
    def __init__(self, data_dir, split='train', tokenizer=None, max_length=512):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.docs_file = os.path.join(data_dir, 'vkmarco-docs.tsv')
        self.queries_file = os.path.join(data_dir, f'vkmarco-doc{split}-queries.tsv')
        self.qrels_file = os.path.join(data_dir, f'vkmarco-doc{split}-qrels.tsv')
        
        self.queries = self._load_queries()
        self.qrels = self._load_qrels()
        
        self.doc_offsets = self._create_doc_index()
        self.length = sum(len(query_docs) for query_docs in self.qrels.values())
        
    def __len__(self):
        return self.length
        
    def _load_queries(self):
        queries = {}
        with open(self.queries_file, 'r') as f:
            for line in f:
                query_id, query_text = line.strip().split('\t')
                queries[query_id] = query_text
        return queries
        
    def _load_qrels(self):
        qrels = defaultdict(list)
        with open(self.qrels_file, 'r') as f:
            for line in f:
                query_id, _, doc_id, relevance = line.strip().split(' ')
                qrels[query_id].append((doc_id, float(relevance)))
        return qrels
        
    def _create_doc_index(self):
        offsets = {}
        with open(self.docs_file, 'r') as f:
            offset = 0
            for line in f:
                doc_id = line.split('\t')[0]
                offsets[doc_id] = offset
                offset += len(line.encode('utf-8'))
        return offsets
        
    def _get_doc_by_id(self, doc_id):
        with open(self.docs_file, 'r') as f:
            f.seek(self.doc_offsets[doc_id])
            line = f.readline()
            _, _, title, body = line.strip().split('\t')
            return title, body
            
    def __iter__(self):
        for query_id, query_docs in self.qrels.items():
            query_text = self.queries[query_id]
            
            for doc_id, relevance in query_docs:
                title, body = self._get_doc_by_id(doc_id)
                doc_text = f"{title} [SEP] {body}"
                
                encoded = self.tokenizer.encode_plus(
                    query_text,
                    doc_text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                yield {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'labels': torch.tensor(relevance / 3.0, dtype=torch.float32)
                }

class RankingModel(nn.Module):
    def __init__(self, model_name='microsoft/mdeberta-v3-base'):
        super(RankingModel, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        for param in self.transformer.encoder.layer[-1].attention.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=False
        )
        
        cls_output = outputs[0][:, 0]
        return self.ranking_head(cls_output).squeeze(-1)

def create_submission(model, tokenizer, data_dir, device, batch_size=128):
    model.eval()
    
    sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    eval_queries = pd.read_csv(
        os.path.join(data_dir, 'vkmarco-doceval-queries.tsv'),
        sep='\t',
        names=['QueryId', 'QueryText']
    )
    
    queries_dict = dict(zip(eval_queries.QueryId, eval_queries.QueryText))
    
    docs_cache = {}
    doc_offsets = {}
    
    with open(os.path.join(data_dir, 'vkmarco-docs.tsv'), 'r') as f:
        offset = 0
        for line in f:
            doc_id = line.split('\t')[0]
            doc_offsets[doc_id] = offset
            offset += len(line.encode('utf-8'))
    
    def get_doc_text(doc_id):
        if doc_id not in docs_cache:
            with open(os.path.join(data_dir, 'vkmarco-docs.tsv'), 'r') as f:
                f.seek(doc_offsets[doc_id])
                line = f.readline()
                _, _, title, body = line.strip().split('\t')
                docs_cache[doc_id] = f"{title} [SEP] {body}"
        return docs_cache[doc_id]
    
    results_dict = {}
    
    with torch.inference_mode():
        for query_id in tqdm(sample_submission['QueryId'].unique()):
            query_text = queries_dict[query_id]
            current_docs = sample_submission[sample_submission['QueryId'] == query_id]['DocumentId'].tolist()
            all_scores = []
            
            for i in range(0, len(current_docs), batch_size):
                batch_docs = current_docs[i:i + batch_size]
                batch_tensors = []
                
                for doc_id in batch_docs:
                    doc_text = get_doc_text(doc_id)
                    
                    encoded = tokenizer(
                        query_text,
                        doc_text,
                        add_special_tokens=True,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    batch_tensors.append({
                        'input_ids': encoded['input_ids'],
                        'attention_mask': encoded['attention_mask']
                    })
                
                batch_input_ids = torch.cat([x['input_ids'] for x in batch_tensors]).to(device)
                batch_attention_mask = torch.cat([x['attention_mask'] for x in batch_tensors]).to(device)
                
                with torch.cuda.amp.autocast():
                    scores = model(batch_input_ids, batch_attention_mask)
                    scores = torch.sigmoid(scores)
                    scores = scores.cpu().numpy()
                
                all_scores.extend(list(zip(batch_docs, scores)))
                
                del batch_input_ids, batch_attention_mask
                torch.cuda.empty_cache()
            
            sorted_pairs = sorted(all_scores, key=lambda x: float(x[1]), reverse=True)
            results_dict[query_id] = [doc_id for doc_id, _ in sorted_pairs]
    
    new_rows = []
    for _, row in sample_submission.iterrows():
        query_id = row['QueryId']
        doc_id = results_dict[query_id].pop(0)
        new_rows.append({'QueryId': query_id, 'DocumentId': doc_id})
    
    submission = pd.DataFrame(new_rows)
    return submission

def train_model(model, train_loader, val_loader, device, num_epochs=1):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=0.01
    )
    
    criterion = nn.MSELoss()
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            desc=f'Epoch {epoch + 1}/{num_epochs}',
            total=train_loader.dataset.length // train_loader.batch_size
        )
        
        for step, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            for param in model.transformer.parameters():
                if not param.requires_grad:
                    param.grad = None
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            train_steps += 1
            
            if step % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            del input_ids, attention_mask, labels, outputs
            
        avg_train_loss = total_loss / train_steps
        
        gc.collect()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Neural ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = RankingModel('xlm-roberta-base')
    
    print("Creating datasets...")
    train_dataset = VKMarcoIterableDataset(
        data_dir=args.data_dir,
        split='train',
        tokenizer=tokenizer
    )
    val_dataset = VKMarcoIterableDataset(
        data_dir=args.data_dir,
        split='dev',
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256,
        pin_memory=True
    )
    
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=3
    )
    
    submission = create_submission(
        model=model,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        device=device
    )
    
    submission.to_csv(args.submission_file, index=False)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")

if __name__ == "__main__":
    main()
