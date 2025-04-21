import torch.nn as nn

# поробовать загрузить токенайзер (navec) в модель, чтобы мозга себе не делать с датасетом и тд

class SentimentClassifier(nn.Module):
    def __int__(self, in_features, out_features):
        super(SentimentClassifier, self).__int__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(self.in_features, self.hidden_size, batch_first=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.drop(x)
        x = self.fc(x)

        return x