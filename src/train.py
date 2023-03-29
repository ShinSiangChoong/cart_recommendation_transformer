import random
import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import ndcg_score

import wandb

from src.cart_dataset import CartDataset
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_items_per_cart', type=int, default=7)
    parser.add_argument('--item_emb_dim', type=int, default=256)
    parser.add_argument('--cat_emb_dim', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=1)

    parser.add_argument('--wandb_mode', type=str, default="disabled")
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./outputs")

    args = parser.parse_args()

    args.cart_path = Path('./input/cart.parquet')
    args.cart_item_path = Path('./input/cart_item.parquet')

    return args


def main(args):
    # read dataframe and define parameters
    df_cart = pd.read_parquet(args.cart_path)
    df_cart_item = pd.read_parquet(args.cart_item_path)
    df_train = df_cart.loc[
        df_cart['test_row_indicator'] == 'TRAIN'
    ].reset_index(drop=True)
    args.n_items = df_cart_item['product_idx'].max()
    args.n_cats = df_cart_item.loc[
        df_train['event_id'].tolist(), 'category_idx'
    ].max()
    args.n_train_items = df_cart_item.loc[
        df_train['event_id'].tolist(), 'product_idx'
    ].max()

    # dataset and dataloader
    train_dataset = CartDataset(
        df_cart=df_train,
        df_cart_item=df_cart_item,
        n_train_items=args.n_train_items,
        max_items_per_cart=args.max_items_per_cart, 
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=False, 
        drop_last=True
    )
    val_dataset = CartDataset(
        df_cart=df_cart.loc[
            df_cart['test_row_indicator'] == 'VAL'
        ].reset_index(drop=True),
        df_cart_item=df_cart_item,
        n_train_items=args.n_train_items,
        max_items_per_cart=None, 
        is_train=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1,
        pin_memory=False, 
        drop_last=False
    )

    # model and paramater initialization
    model = Model(
        args.n_items, args.n_cats, args.item_emb_dim, args.cat_emb_dim
    )
    model.cuda()
    num_train_optimization_steps = (
        args.epochs * len(train_loader) // args.accumulation_steps
    )
    warmup_steps = len(train_loader) // args.accumulation_steps
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='none')   

    # training
    scaler = torch.cuda.amp.GradScaler()
    state_dict = {}
    best_ndcg = 0
    non_improving_epoch = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        avg_loss = 0
        tbar = tqdm(train_loader, total=len(train_loader))
        for i, (src, cat, _, target, att_mask) in enumerate(tbar):
            src, cat, target, att_mask = (
                src.cuda(), cat.cuda(), target.cuda(), att_mask.cuda()
            )
            with torch.cuda.amp.autocast():
                out = model(src, cat, att_mask)
                loss = criterion(out.permute(0, 2, 1), target)
                loss = torch.mean(loss)
                scaler.scale(loss).backward()
                avg_loss += loss.item()
                
            if i % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss /= (i+1)
        
        model.eval()
        outs = []
        targets = []
        tbar = tqdm(val_loader, total=len(val_loader))
        with torch.inference_mode():
            for i, (src, cat, mask, target, att_mask) in enumerate(tbar):
                src, cat, mask, target, att_mask = (
                    src.cuda(), 
                    cat.cuda(), 
                    mask.cuda(), 
                    target.cuda(), 
                    att_mask.cuda()
                )
                with torch.cuda.amp.autocast():
                    out = model(src, cat, att_mask)
                    idx = torch.where(~mask)
                    y_true = F.one_hot(target[idx], num_classes=args.n_items)
                    outs.append(out[idx].detach().cpu().numpy())
                    targets.append(y_true.detach().cpu().numpy())
        y_true = np.concatenate(targets)
        y_pred = np.concatenate(outs)
        ndcg = ndcg_score(y_true, y_pred)
        print(f"epoch{epoch}:", ndcg)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            non_improving_epoch = 0
            # save best model
            state_dict.update({
                'weights': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'val_ndcg': ndcg,
                'train_loss': avg_loss
            })
            torch.save(state_dict, f"{args.output_dir}/best_model.bin")
        else:
            non_improving_epoch += 1
        
        # early stopping
        if non_improving_epoch >= 3:
            break 


if __name__ == '__main__':
    args = parse_args()
    wandb.init(
        project="AI4Code Nested Transformer",
        name=args.wandb_name,
        mode=args.wandb_mode,
    )
    try:
        main(args)
    finally:
        wandb.finish()