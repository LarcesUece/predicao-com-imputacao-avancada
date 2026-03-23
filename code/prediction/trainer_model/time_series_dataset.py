import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """
        Time Series Dataset for PyTorch
        --------------------------------
        This class creates a dataset for time series data to be used in PyTorch models.
        
        Args:
            data (array-like): The time series data.
            window_size (int): The size of the input window.
            target_col (int): The index of the target column in the data.
        Returns:
            Dataset object for time series data.
    """
    def __init__(self, data, window_size, target_col=0):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.target_col = target_col
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size, self.target_col]
        return x, y