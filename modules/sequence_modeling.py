import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class ResTrans(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(512, 4)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=4)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512, out_features=output_size)

    def forward(self, x):
        # batch_size x T x input_size
        batch_size, channel, size = x.shape
        x = self.transformer(x)
        x = self.pooling(x.permute(0, 2, 1)).view(batch_size, 512)
        x = self.linear(x).unsqueeze(1)
        return x
