import torch
import random
import argparse
import time
import scipy.io
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import ChaosDataset
from timm.scheduler import create_scheduler
from timm.scheduler import CosineLRScheduler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from model import CNN_ChaosClassifier, init_weights, LSTM_ChaosClassifier, Transformer_ChaosClassifier

if __name__ == '__main__':

    # Init
    lr_rate = 0.001
    num_epochs = 200
    batch_size = 16

    max_len = 5001
    train_data_name = "train_5004"
    test_data_name = "test_5004"
    model_name = 'Transformer_ChaosClassifier'   # CNN_ChaosClassifier  LSTM_ChaosClassifier  Transformer_ChaosClassifier

    if model_name == 'CNN_ChaosClassifier':
        checkpoint_path = "checkpoint/" + model_name + "_" + str(max_len) + ".pth"
    elif model_name == 'LSTM_ChaosClassifier':
        checkpoint_path = "checkpoint/" + model_name + "_" + str(max_len) + ".pth"
    elif model_name == 'Transformer_ChaosClassifier':
        checkpoint_path = "checkpoint/" + model_name + "_" + str(max_len) + ".pth"
    else:
        raise NotImplementedError

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data
    train_data = scipy.io.loadmat('data/' + train_data_name + '.mat')['data']
    train_x = train_data[:, 4:]
    train_y = train_data[:, 3]
    test_data = scipy.io.loadmat('data/' + test_data_name + '.mat')['data']
    test_x = test_data[:, 4:]
    test_y = test_data[:, 3]

    # norm
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # dataset
    train_dataset = ChaosDataset(train_x, train_y)
    test_dataset = ChaosDataset(test_x, test_y)

    # dataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model  1. CNN_ChaosClassifier  2. LSTM_ChaosClassifier  3. Transformer_ChaosClassifier
    if model_name == 'CNN_ChaosClassifier':
        model = CNN_ChaosClassifier(input_dim=1, hidden_dim=64, kernel_size=3, stride=1, padding=2).to(device)
        model.apply(init_weights)
    elif model_name == 'LSTM_ChaosClassifier':
        model = LSTM_ChaosClassifier(input_size=1, hidden_size=64, num_classes=2).to(device)
        model.apply(init_weights)
    elif model_name == 'Transformer_ChaosClassifier':
        model = Transformer_ChaosClassifier(input_dim=1, hidden_dim=32, nhead=2, num_layers=2, num_classes=2, device=device, max_len=max_len).to(device)
        model.apply(init_weights)
    else:
        raise NotImplementedError

    # loss & optimizer & lr_scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # train
    best_accuracy = 0.0
    total_time = 0
    for epoch in range(num_epochs):
        model.train()
        loss_dict = 0.0
        train_correct, train_total = 0, 0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss
            loss_dict += loss.item()
            _, predicted = torch.max(outputs, 1)

            # train acc
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # time
        end_time = time.time()
        total_time = total_time + (end_time - start_time)

        # update scheduler
        current_lr = optimizer.param_groups[0]["lr"]

        # eval
        model.eval()
        test_correct, test_total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # test acc
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total

        # save checkpoint
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
            }, checkpoint_path)

        print(f"Epoch {epoch + 1}, Lr_rate {current_lr:.6f}, Loss: {loss_dict:.4f}, Train Accuracy: {train_acc:.2f}%, "
              f"Test Accuracy: {test_acc:.2f}%, Best Accuracy: {best_accuracy:.2f}%, Time: {total_time:.2f} s")

    print("Training is completed!")
