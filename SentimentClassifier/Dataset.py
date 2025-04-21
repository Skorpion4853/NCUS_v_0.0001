import re
import torch
import torch.utils.data as data

class SentimentDataset(data.Dataset):
    def __init__(self, path, navec_emb, prev_words=3):
        self.prev_words = prev_words
        self.navec_emb = navec_emb

        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\ufeff', '')  # убираем первый невидимый символ
            self.text = self.text.replace('\n', ' ')
            self.text = re.sub(r'[^А-яA-z- ]', '', self.text)  # удаляем все неразрешенные символы

        self.words = self.text.lower().split()
        self.words = [word for word in self.words if word in self.navec_emb]  # оставляем слова, которые есть в словаре
        vocab = set(self.words)

        self.int_to_word = dict(enumerate(vocab))
        self.word_to_int = {b: a for a, b in self.int_to_word.items()}
        self.vocab_size = len(vocab)

    def __getitem__(self, item):
        _data = torch.vstack([torch.tensor(self.navec_emb[self.words[x]]) for x in range(item, item+self.prev_words)])
        word = self.words[item+self.prev_words]
        t = self.word_to_int[word]

        return _data, t

    def __len__(self):
        return len(self.words) - 1 - self.prev_words