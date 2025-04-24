import re
import pandas as pd
import torch
import torch.utils.data as data
import numpy as np

class SentimentDataset(data.Dataset):
    def __init__(self, path, navec_emb):
        self.navec_emb = navec_emb

        self.data = pd.read_csv(path)
        self.len = len(self.data)

    def __getitem__(self, item):
        self.data.iloc[item, 0:1] = re.sub(r'[^А-яA-z- ]', '',
                                 str(self.data.iloc[item, 0]).replace('\ufeff', '').replace('\n', ' ')).lower()
        words = [word for word in str(self.data.iloc[item, 0:1]).split(' ') if word in self.navec_emb]
        text = torch.vstack([torch.tensor(self.navec_emb[word]) for word in words])
        '''
        out_text = torch.zeros(128, 300)
        out_text[..., :len(words), :300] = text
        '''


        label = torch.from_numpy(self.data.iloc[item, 1:29].to_numpy(dtype='float32'))
        return text, label

    def __len__(self):
        return self.len