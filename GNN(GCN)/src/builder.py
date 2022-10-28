'''
Build the bipartie graph from csv
'''
#Load Package
import torch
import torch.nn as nn
import numpy as np
import torch_geometric
import pandas as pd
from sentence_transformers import *
from config import DefaultConfig
from torch_geometric.data import HeteroData
from builder_utils import load_node_csv, load_edge_csv, SequenceEncoder, SameEncoder

# Get the path for each dataset
config = DefaultConfig()

# Get product information
product_x, product_mapping = load_node_csv(config.product_path, index_col='product_id',
                                           encoders={'brand': SequenceEncoder(), 'cat_0': SequenceEncoder(),
                                                     'cat_1': SequenceEncoder(),
                                                     'price': SameEncoder(dtype=torch.long)})
user_x, user_mapping = load_node_csv(config.user_path, index_col='user_id')

season_x, season_mapping = load_node_csv(config.season_path, index_col = 'ts_month',
                                         encoders = {'ts_weekday': SameEncoder(dtype = torch.long),
                                                     'ts_day' : SameEncoder(dtype = torch.long)})

# Build the graph object
# Set the two nodes: product and user
data = HeteroData()
data['user'].num_nodes = len(user_mapping)
data['product'].x = product_x
data['season'].x = season_x

# Set the edge for nodes
edge_index, edge_label = load_edge_csv(config.edge_path, 'user_id',user_mapping,'product_id', product_mapping)
data['user','like','product'].edge_index =  edge_index

us_index, us_label = load_edge_csv(config.us_edge_path, 'user_id', user_mapping, 'ts_month', season_mapping)
data['user', 'buy', 'season'].edge_index = us_index

sp_index, sp_label = load_edge_csv(config.sp_edge_path, 'ts_month', season_mapping, 'product_id', product_mapping)
data['season', 'popular', 'product'].edge_index = sp_index

# save the graph
output = {'graph':data}
torch.save(output, config.graph_output_path)


