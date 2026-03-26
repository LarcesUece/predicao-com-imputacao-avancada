import torch.nn as nn


class RNNForecaster(nn.Module):
    """
    RNN Forecaster model
    ----------------------
    This class will execute for all files in the directory 'imputed_series_selective_unfair',
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in RNN
        num_layers (int): Number of RNN layers
        rnn_type (str): Type of RNN ('LSTM' or 'GRU')
        dropout (float): Dropout rate
    Returns:
        output (Tensor): Predicted values
    """

    def __init__(
        self, input_size=1, hidden_size=64, num_layers=2, rnn_type="LSTM", dropout=0.1
    ):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(1)


class RNNModel(nn.Module):
    """
    RNN Model for time series prediction.
    ---------------------------------
    This model will be used for filtered links with the aim of improving the RMSE values
    and the improvement_pct column of the stacking methods compared to the baseline methods.
    This class will be used in the 'improvement_filtered_links.py' file.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the RNN.
        num_layers (int): Number of RNN layers.
        model_type (str): Type of RNN ('LSTM' or 'GRU').
        dropout (float): Dropout rate.
    Returns:
        output (Tensor): Predicted values.
    """

    def __init__(
        self, input_size, hidden_size, num_layers, model_type="LSTM", dropout=0.1
    ):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type

        if model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif model_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.squeeze(-1)
