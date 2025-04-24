import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(self.in_features, self.hidden_size, batch_first=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.drop(h[-1])
        y = self.fc(y)

        return y

class SentimentClassifierGRU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.GRU(self.in_features, self.hidden_size, batch_first=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.drop(h[-1])
        y = self.fc(y)

        return y