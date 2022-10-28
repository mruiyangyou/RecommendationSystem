'''
Building underltying GNN MODEL and prediction layer
'''

import torch
import torch.nn as nn
import numpy as np
import torch_geometric
import pandas as pd
from sentence_transformers import *
from torch_geometric.data import HeteroData

# covert node csv to graph data
def load_node_csv(path, index_col, encoders=None):
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = None

    if encoders:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

# Encode sequence data
class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True)

        return x.cpu()

# Encoder data
class SameEncoder:

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

# Convert edge csv to graph
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
    df = pd.read_csv(path)
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]

    edge = torch.tensor([src, dst])
    edge_attr = None
    if encoders:
        edge_att = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_att, dim=-1)

    return edge, edge_attr
