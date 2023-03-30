import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class CartDataset(Dataset):
    """ 
    Pytorch Dataset to represent the items in the carts
    Unknown/padded items/categories are represented using 0
    Args:
        df_cart (pd.DataFrame): a cart level dataframe
        df_cart_item (pd.DataFrame): a cart-item pair dataframe
        n_train_items (int): number of item for training
        max_items_per_cart (int): maximum number of items per cart
        is_train (bool): training if true else evaluating
    """
    def __init__(
        self, 
        df_cart, 
        df_cart_item, 
        n_train_items,
        max_items_per_cart, 
        is_train=True, 
    ):
        self.df_cart = df_cart
        self.df_cart_item = df_cart_item
        self.max_items_per_cart = max_items_per_cart
        self.n_train_items = n_train_items
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df_cart)

    def __getitem__(self, idx):
        event_id = self.df_cart['cart_id'].iloc[idx]
        item_cats = self.df_cart_item.loc[
            event_id, ['product_idx', 'category_idx']
        ]
        if self.is_train and len(item_cats) > self.max_items_per_cart:
            item_cats = item_cats.sample(self.max_items_per_cart)
        items = item_cats['product_idx'].tolist()
        cats = item_cats['category_idx'].tolist()
        
        n_items = len(items)
        mask = [1]*n_items
        if self.is_train:
            mask_idx = np.random.choice(n_items) # mask a random item
            # mask_idx = n_items - 1  # mask only last item
            pad_len = self.max_items_per_cart - n_items
            att_mask = [0]*n_items + [1]*pad_len
            items += [0]*pad_len
            cats += [0]*pad_len
            mask += [0]*pad_len
        else:
            att_mask = [0]*len(items)
            # mask_idx = idx % n_items  # mask deterministically for evaluation 
            mask_idx = n_items - 1  # mask last item for evaluation
            
        mask[mask_idx] = 0
        mask = torch.BoolTensor(mask)
        src = torch.LongTensor(items) * mask
        cats = torch.LongTensor(cats) * mask
        target = torch.clip(torch.LongTensor(items) - 1, min=0)
        att_mask = torch.BoolTensor(att_mask)
        
        # use 0 for items not in the training set
        if not self.is_train:
            src *= (src <= self.n_train_items) 
            
        return src, cats, mask, target, att_mask
    
    
class CartDatasetProd(Dataset):
    """ 
    Pytorch Dataset to represent the items in the carts (for production)
    Unknown/padded items/categories are represented using 0
    Items/Categories that do not exist during training are treated as unknowns
    Args:
        df_cart (pd.DataFrame): a cart level dataframe
        df_cart_item (pd.DataFrame): a cart-item pair dataframe
    """
    def __init__(
        self, 
        df_cart, 
        df_cart_item, 
    ):
        self.df_cart = df_cart
        self.df_cart_item = df_cart_item
        
    def __len__(self):
        return len(self.df_cart)

    def __getitem__(self, idx):
        event_id = self.df_cart['cart_id'].iloc[idx]
        item_cats = self.df_cart_item.loc[
            event_id, ['product_idx', 'category_idx']
        ]

        items = item_cats['product_idx'].tolist() + [0]
        cats = item_cats['category_idx'].tolist() + [0]

        src = torch.LongTensor(items)
        cats = torch.LongTensor(cats)
        att_mask = torch.BoolTensor([True] * len(items))
        
        return src, cats, att_mask  