import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import scipy

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import ndcg_score, average_precision_score

from tqdm import tqdm

from cart_dataset import CartDataset
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument(
        '--weight_path', type=str, default="./outputs/best_model.bin"
    )
    args = parser.parse_args()

    args.cart_path = Path('./input/cart.parquet')
    args.cart_item_path = Path('./input/cart_item.parquet')

    return args

def main(args):
    # read data
    df_cart = pd.read_parquet(args.cart_path)
    df_cart_item = pd.read_parquet(args.cart_item_path)
    df_cart_test = df_cart.loc[
        (df_cart['test_row_indicator'] == 'TEST')
    ].reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.weight_path, map_location='cpu')

    # dataset and dataloader
    test_dataset = CartDataset(
        df_cart=df_cart_test,
        df_cart_item=df_cart_item,
        n_train_items=state_dict['n_train_items'],
        max_items=None,
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1,
        pin_memory=False, 
        drop_last=False
    )

    # model initialization
    model = Model(
        state_dict['n_items'], 
        state_dict['n_cats'], 
        state_dict['item_emb_dim'], 
        state_dict['cat_emb_dim']
    ).to(device)
    model.load_state_dict(state_dict['weights'])
    model.eval()

    # inference
    outs = []
    targets = []
    tbar = tqdm(test_loader, total=len(test_loader))
    with torch.inference_mode():
        for i, (src, cat, mask, target, att_mask) in enumerate(tbar):
            src, cat, mask, target, att_mask = (
                src.to(device), 
                cat.to(device), 
                mask.to(device), 
                target.to(device), 
                att_mask.to(device)
            )
            with torch.cuda.amp.autocast():
                out = model(src, cat, att_mask)
                idx = torch.where(~mask)
                y_true = F.one_hot(
                    target[idx], num_classes=state_dict['n_items']
                )
                outs.append(out[idx].detach().cpu().numpy())
                targets.append(y_true.detach().cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(outs)
    # y_pred_all = scipy.special.softmax(y_pred_all, axis=-1)

    # evaluate
    ndcg = ndcg_score(y_true, y_pred)
    aps = []
    for y, p in zip(y_true, y_pred):
        aps.append(average_precision_score(y, p))
    mean_ap = np.mean(aps)
    print('ndcg:', ndcg, 'mAP:', mean_ap)


if __name__ == '__main__':
    args = parse_args()
    main(args)