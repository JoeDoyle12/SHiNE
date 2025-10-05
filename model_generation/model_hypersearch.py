from model_traintest import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_factory import build_ttvtdv_model
import optuna
import numpy as np


def make_objective(train_loader, val_loader):
    def objective(trial):
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128, 256])
        n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2, 3)
        transformer_heads = trial.suggest_categorical("transformer_heads", [2, 4, 8])
        transformer_layers = trial.suggest_int("transformer_layers", 1, 2, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.2, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-3, 1e-2)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_ttvtdv_model(
            lstm_hidden=lstm_hidden,
            n_lstm_layers=n_lstm_layers,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
            device=device
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_epochs = 10

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
            trial.report(val_accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_loss
    
    return objective
