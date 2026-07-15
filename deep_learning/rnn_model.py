import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNNClassifier, self).__init__()
        
        # 1. Embedding Layer: Converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # bidirectional=True doubles the hidden dimension for the final layer
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 3. Fully Connected Layer: Maps RNN output to class scores
        # If bidirectional, we need hidden_dim * 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, sent len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch size, sent len, emb dim]

        output, hidden = self.rnn(embedded)        # output shape: [batch size, sent len, hid dim * num directions]
        # hidden shape: [num layers * num directions, batch size, hid dim]
        
        # Concatenate the final forward and backward hidden states if bidirectional
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = self.dropout(hidden)
        else:
            hidden = hidden[-1]
            hidden = self.dropout(hidden)
        # hidden shape: [batch size, hid dim * num directions]
        
        return self.fc(hidden)
