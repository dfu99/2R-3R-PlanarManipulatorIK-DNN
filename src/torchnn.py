from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader

# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device {DEVICE}.")

class RobotNN(nn.Module):
    def __init__(self,num_inputs,num_outputs,hidden_layer_size=64):
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(num_inputs, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_outputs),
        )

    def forward(self, x):
        logits = self.linear_relu(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y)
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


class CustomTorchData(Dataset):
    """
    Creates a PyTorch DataLoader-friendly dataset from your input/output data
    """
    def __init__(self, inputs, outputs):
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def data2loader(dataInput, dataOutput, training_split):
    data_size = len(dataInput)
    split_idx = int(data_size * training_split)
    train_x = dataInput[:split_idx, :]
    train_y = dataOutput[:split_idx, :]
    test_x = dataInput[split_idx:, :]
    test_y = dataOutput[split_idx:, :]

    # Create a dataloader from the data
    train_dataset = CustomTorchData(train_x, train_y)
    test_dataset = CustomTorchData(test_x, test_y)
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader