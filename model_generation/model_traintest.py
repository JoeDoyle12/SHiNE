import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model_factory import build_ttvtdv_model
from data_collector import get_data
import numpy as np

class TransitDataset(Dataset):
    def __init__(self, X, y, sequence_length=135):
        """
        X: numpy or torch tensor of shape (N, seq_len, num_features)
        y: numpy or torch tensor of shape (N,)
        """
        # print(len(X))
        # tensors = [torch.tensor(x, dtype=torch.float32) for x in X]
        # self.X = torch.tensor(pad_sequence(tensors, batch_first=True, padding_value=0.0))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, train_loader, val_loader, criterion, optimizer, device="cpu"):
    model.train() # tells the model that I am training
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs, _ = model(X)
        loss = criterion(outputs.squeeze(), y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader) # return average loss

def evaluate_model(model, val_loader, criterion, device="cpu"):
    model.eval()
    # with torch.zero_grad(): # Stops gradients from being calculated, saves memory and speeds it up
    val_loss = 0
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        outputs, _ = model(X)
        val_loss += criterion(outputs.squeeze(1), y.float()).item() * X.size(0)
        probs = torch.sigmoid(outputs)
        print(probs)
        print(
        "pred.mean", outputs.mean().item(),
        "pred.std",  outputs.std().item(),
        "head_w_norm", sum(p.norm().item() for p in model.classifier.parameters() if p.requires_grad),
        )

        preds = torch.round(outputs) # rounds to 0 or 1

        # Count how many 1's were predicted correctly
        true_positives = ((preds == 1) & (y == 1)).sum().item()

        p_tp = true_positives / y.sum().item()
    
    return val_loss / len(val_loader), p_tp # return validation loss and percent true positives


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    y_train = torch.cat([y for _, y in train_loader], dim=0)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_w = torch.tensor([neg / max(pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    model.to(device)

    for epoch in range(epochs):
        # Training
        avg_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer, device)

        # Validation
        val_loss, p_tp = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Percent True Positives: {p_tp:.4f}%")

    return model


def test_model(model, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()

    test_loss, p_tp = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f} | Test Percent True Positives: {p_tp:.4f}%")


if __name__ == '__main__': # simple test of training capabilities
    import numpy as np

    seq_len = 135  # number of transits
    num_features = 2  # TTV + TDV


    X, y = get_data()

    N = len(X)   # number of samples

    # mask = torch.zeros(N, seq_len, 2, dtype=torch.bool)

    # for i in range(N):
    #     for j in range(seq_len):
    #         if X[i][j][0] != 0 or X[i][j][1] != 0:
    #             continue
    #         mask[i, j, 0] = True
    #         mask[i, j, 1] = True

    print(y)
    # Train/val split
    val_split = int(0.8 * N)
    test_split = int(0.9 * N)
    X_train, y_train = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:test_split], y[val_split:test_split]
    X_test, y_test = X[test_split:], y[test_split:]

    print(len(X_train), len(X_val), len(X_test))
    print(X_train)

    train_data = TransitDataset(X_train, y_train)
    val_data = TransitDataset(X_val, y_val)
    test_data = TransitDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    # Instantiate model
    model = build_ttvtdv_model(input_size=2, lstm_hidden=64, transformer_d_model=128,
                               device="cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # Train
    trained_model = train_model(model, train_loader, val_loader,
                                epochs=50, lr=1e-3,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    
    test_model(trained_model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu")
