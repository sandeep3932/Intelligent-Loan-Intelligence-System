import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout
    ):
        super(LSTMClassifier, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Fully Connected Layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text -> [batch_size, sequence_length]

        embedded = self.dropout(self.embedding(text))

        # output -> [batch_size, seq_len, hidden_dim * directions]
        # hidden -> [layers * directions, batch_size, hidden_dim]
        # cell   -> [layers * directions, batch_size, hidden_dim]

        output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)

        return self.fc(hidden)