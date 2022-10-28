import time
import os
class DefaultConfig:

    path = '/Users/marceloyou/Desktop/算法/RecommanationSystem/GNN(GCN)/data/'

    # data path
    product_path = os.path.join(path, 'train_product_3.csv')
    user_path = os.path.join(path, 'train_user_3.csv')
    edge_path = os.path.join(path, 'train_edge_3.csv')
    season_path = os.path.join(path, 'train_season_3.csv')
    us_edge_path = os.path.join(path, 'train_edge(user, season)_3.csv')
    sp_edge_path = os.path.join(path, 'train_edge(season,product)_3.csv')

    # model path
    output_name = time.strftime('graph_' + '%m%d_%H:%M:%S.pth')
    graph_output_path = os.path.join(path, output_name)
