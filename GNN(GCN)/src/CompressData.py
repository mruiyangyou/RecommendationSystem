import numpy as np
import pandas as pd
import os

# get the list of fils
path = '/Users/marceloyou/Desktop/算法/RecommanationSystem/GNN(GCN)/data/'
filelist = os.listdir(path)
filelist = list(filter(lambda x: 'parquet' in x, filelist))

# Preprocess on data
def process_df(path):
    df = pd.read_parquet(path)
    user = df['user_id'].unique()
    np.random.seed(42)
    sample_user = np.random.choice(user, 100)
    for i in df.columns:
        df.loc[df[i] == 'NA', i] = np.nan
    df.drop(columns=['cat_3'], inplace=True)
    condition = (df['brand'].isnull())|(df['cat_2'].isnull())
    df = df.loc[-condition] 
    df = df.loc[df['user_id'].isin(sample_user)]
    df['ts_weekday'] = df['ts_weekday'].apply(lambda x: 0 if x <= 4 else 1)
    df['ts_day'] = df['ts_day'].apply(lambda x: 0 if x <= 10 else 1 if 10 < x <= 20 else 2)
    df['ts_day'] = df['ts_day'].astype('category')
    df['ts_weekday'] = df['ts_weekday'].astype('category')
    df.reset_index(inplace=True)
    print('Dataframe shape:', df.shape, '\n', 'Dataframe NA:\n', df.isnull().sum())
    return df

# Load relevant data and prepross it
train = process_df(os.path.join(path, filelist[0]))
# valid = process_df(os.path.join(path, filelist_3[1]))
# test = process_df(os.path.join(path, filelist_3[2]))

# Save the csv
train[['product_id','price','brand','cat_0','cat_1']].to_csv(os.path.join(path, 'train_product_3.csv'))
train['user_id'].to_csv(os.path.join(path, 'train_user_3.csv'))
train[['user_id','product_id']].to_csv(os.path.join(path, 'train_edge_3.csv'))
train[['user_id', 'ts_month']].to_csv(os.path.join(path, 'train_edge(user, season)_3.csv'))
train[['ts_month', 'product_id']].to_csv(os.path.join(path, 'train_edge(season,product)_3.csv'))
train[['ts_weekday', 'ts_day','ts_month']].to_csv(os.path.join(path, 'train_season_3.csv'))