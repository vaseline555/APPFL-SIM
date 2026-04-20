import torch


class SmartMeterLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        hidden_size_2=16,
        num_layers=2,
        num_classes=1,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.hidden_size_2 = int(hidden_size_2)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)

        # PyTorch LSTM uses tanh activations internally, matching the requested setup.
        self.lstm1 = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size_2,
            num_layers=max(1, self.num_layers - 1),
            batch_first=True,
        )
        self.head = torch.nn.Linear(self.hidden_size_2, self.num_classes)

    def forward(self, x):
        if x.ndim == 1:
            x = x.view(1, -1, 1)
        elif x.ndim == 2:
            x = x.unsqueeze(-1)

        x, _ = self.lstm1(x.float())
        x, _ = self.lstm2(x)
        return self.head(x[:, -1, :])
