import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """ 
    A transformer model for completing item sets
    Args:
        n_items (int): number of unique items (excluding unknowns)
        n_cats (int): number of unique item categories (excluding unknowns)
        item_emb_dim (int): item embedding dimension
        cat_emb_dim (int): category embedding dimension
    """
    def __init__(self, n_items, n_cats, item_emb_dim, cat_emb_dim):
        super(Model, self).__init__()
        self.item_emb = nn.Embedding(n_items+1, item_emb_dim)
        self.cat_emb = nn.Embedding(n_cats+1, cat_emb_dim)
        emb_dim = item_emb_dim+cat_emb_dim
        self.tfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim, 
                nhead=8, 
                batch_first=True
            ), 
            num_layers=6
        )
        self.linear = nn.Linear(emb_dim, n_items)
        
    def forward(self, src, cats, att_mask):
        src = self.item_emb(src)
        cats = self.cat_emb(cats)
        x = self.tfm(
            torch.concat([src, cats], axis=-1), 
            src_key_padding_mask=att_mask, 
        )
        return self.linear(x)