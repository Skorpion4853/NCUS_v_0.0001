import torch
import pandas as pd
import re
from navec import Navec
from slovnet.model.emb import NavecEmbedding
from sklearn.model_selection import train_test_split

'''
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

a = navec['арбуз']

print(f'\n')
'''

'''
df = pd.read_parquet("hf://datasets/seara/ru_go_emotions/raw/train-00000-of-00001-86de8ef1d0ae28df.parquet")

df.to_csv('train.csv', index=False)
'''

'''
#ru_text,text,id,,admiration,amusement,anger,annoyance,approval,caring,confusion,curiosity,desire,disappointment,disapproval,disgust,embarrassment,excitement,fear,gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral

df = pd.read_csv('train.csv')

df.drop(columns=['text','id','author','subreddit','link_id','parent_id','created_utc','rater_id','example_very_unclear'], inplace=True)

df.to_csv('train_.csv', index = False)

#Я ЗАБИЛ ОГРОМНЫЙ на уменьшение labels, крч, будет 28 эмоций, мне поххххххх
'''

'''
df = pd.read_csv('datasets/train_.csv')

for i in range(len(df)):
    df.iloc[i, 0:1] = re.sub(r'[^А-яA-z- ]', '', str(df.iloc[i, 0]).replace('\ufeff', '').replace('\n', ' ')).lower()

df.to_csv('train.csv',index=False)
'''

df = pd.read_csv('datasets/data.csv')

train, val = train_test_split(df,
                              test_size=0.2)

print('a')