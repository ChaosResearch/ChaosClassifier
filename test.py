import torch
import random
import scipy.io
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import ChaosDataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from model import CNN_ChaosClassifier, init_weights, LSTM_ChaosClassifier, Transformer_ChaosClassifier
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    # Init
    batch_size = 16
    max_len = 5001
    test_data_name = "2D_ceshi1_2"
    model_name = 'CNN_ChaosClassifier'  # CNN_ChaosClassifier  LSTM_ChaosClassifier  Transformer_ChaosClassifier

    if model_name == 'CNN_ChaosClassifier':
        checkpoint_path = "checkpoint/" + model_name + "_" + str(max_len) + ".pth"
    elif model_name == 'LSTM_ChaosClassifier':
        checkpoint_path = "checkpoint/" + model_name + "_" + str(max_len) + ".pth"
    elif model_name == 'Transformer_ChaosClassifier':
        checkpoint_path = "checkpoint/" + model_name + "_" + str(max_len) + ".pth"
    else:
        raise NotImplementedError

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data
    test_data = scipy.io.loadmat('data/' + test_data_name + '.mat')['data']
    test_x = test_data[:, 4:]
    test_y = test_data[:, 3]

    # norm
    scaler = StandardScaler()
    test_x = scaler.fit_transform(test_x)

    # dataset
    test_dataset = ChaosDataset(test_x, test_y)

    # dataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    if model_name == 'CNN_ChaosClassifier':
        model = CNN_ChaosClassifier(input_dim=1, hidden_dim=64, kernel_size=3, stride=1, padding=2).to(device)
        model.apply(init_weights)
    elif model_name == 'LSTM_ChaosClassifier':
        model = LSTM_ChaosClassifier(input_size=1, hidden_size=64, num_classes=2, dropout=0.3).to(device)
        model.apply(init_weights)
    elif model_name == 'Transformer_ChaosClassifier':
        model = Transformer_ChaosClassifier(input_dim=1, hidden_dim=32, nhead=2, num_layers=2, num_classes=2,
                                            device=device, max_len=max_len).to(device)
        model.apply(init_weights)
    else:
        raise NotImplementedError

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # test
    model.eval()
    predictions = []
    true_labels = []
    test_correct, test_total = 0, 0
    chaos_correct, chaos_total = 0, 0
    rules_correct, rules_total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

            # overall test acc
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # rule test acc
            rules_mask = (labels == 1)
            rules_total += rules_mask.sum().item()
            rules_correct += (predicted[rules_mask] == labels[rules_mask]).sum().item()

            # chaos test acc
            chaos_mask = (labels == 0)
            chaos_total += chaos_mask.sum().item()
            chaos_correct += (predicted[chaos_mask] == labels[chaos_mask]).sum().item()

    test_data = np.insert(test_data, 4, predictions, axis=1)
    test_data = test_data[:, :5]

    test_acc = 100 * test_correct / test_total
    rules_acc = 100 * rules_correct / rules_total if rules_total > 0 else 0
    chaos_acc = 100 * chaos_correct / chaos_total if chaos_total > 0 else 0
    f1 = f1_score(true_labels, predictions, average='binary')

    print(f"{model_name} test accuracy: {test_acc:.2f}%")
    print(f"Rules sequence accuracy: {rules_acc:.2f}%")
    print(f"Chaos sequence accuracy: {chaos_acc:.2f}%")
    print(f"F1 score: {f1*100:.2f}%")

