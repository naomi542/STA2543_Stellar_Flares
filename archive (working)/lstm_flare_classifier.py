"""
lstm_flare_classifier.py

Train or load an LSTM model with attention to classify each timestep in a light curve
as one of three flare phases: 0 (no flare), 1 (rise), 2 (decay).

Data should come from flare-injected synthetic light curves with:
- time
- synthetic_flux
- flare_phase_labels

Usage:
    from lstm_flare_classifier import train_model, load_model, evaluate_model, predict_single_sequence
    model = train_model(dataset_path='..data/synthetic_lightcurves.pkl')
    model = load_model()
    evaluate_model(model, dataset_path='..data/synthetic_lightcurves.pkl')
    prediction = predict_single_sequence(model, flux_sequence)
"""

import os
import json
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

# ----------------------------
# Config
# ----------------------------
SEQ_LEN = 200
STRIDE= 50
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
SAVE_PATH = "../models/lstm_flare_classifier.pt"
SPLIT_PATH = "../models/splits.json"
SYNTHETIC_PATH= "../data/synthetic_lightcurves.pkl"

# ----------------------------
# Dataset Class (Star-based Splits)
# ----------------------------
class FlarePhaseDataset(Dataset):
    def __init__(self, lightcurve_dict, seq_len, tic_ids, stride=STRIDE, require_flare=False):
        self.samples = []
        for tic in tic_ids:
            entry = lightcurve_dict[tic]
            flux = entry['synthetic_flux']
            labels = entry['flare_phase_labels']
            for i in range(0, len(flux) - seq_len, stride):
                label_seq = labels[i:i+seq_len]
                if require_flare and not np.any(np.array(label_seq) > 0):
                    continue
                self.samples.append((flux[i:i+seq_len], label_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flux_seq, label_seq = self.samples[idx]
        flux_seq = torch.tensor(flux_seq, dtype=torch.float32).unsqueeze(-1)
        label_seq = torch.tensor(label_seq, dtype=torch.long)
        return flux_seq, label_seq

# ----------------------------
# LSTM Model with Attention
# ----------------------------
class LSTMFlareClassifier(nn.Module):
    """
    LSTM-based model with attention mechanism to classify each timestep
    in a light curve sequence into one of the flare phases.

    Parameters:
        input_size (int): Size of the input feature vector (default: 1).
        hidden_size (int): Number of LSTM hidden units.
        num_layers (int): Number of LSTM layers.
        num_classes (int): Number of flare phase classes (default: 3).
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        attended = lstm_out * attn_weights
        out = self.fc(attended)
        return out

# ----------------------------
# Train Model
# ----------------------------

# Compute class weights dynamically
def compute_class_weights(dataset):
    """Compute class weights from a dataset of (seq, label_seq) samples"""
    from collections import Counter
    label_counts = Counter()
    for _, label_seq in dataset:
        label_counts.update(label_seq.numpy().tolist())

    total = sum(label_counts.values())
    weights = [total / label_counts.get(i, 1) for i in range(3)]
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Computed class weights: {weights}")
    return weights

def train_model(dataset_path=SYNTHETIC_PATH, split_path= SPLIT_PATH, model_path= SAVE_PATH):
    """
    Trains the LSTM model on synthetic flare phase data using star-based splitting.

    Parameters:
        dataset_path (str): Path to the pickled dictionary of processed synthetic light curves.

    Returns:
        model (nn.Module): Trained LSTMFlareClassifier model.
    """
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} stars from {dataset_path}")

    all_tics = list(data.keys())
    train_val_tics, test_tics = train_test_split(all_tics, test_size=0.1, random_state=42)
    train_tics, val_tics = train_test_split(train_val_tics, test_size=0.2222, random_state=42)

    splits = {"train": train_tics, "val": val_tics, "test": test_tics}
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump(splits, f)

    train_ds = FlarePhaseDataset(data, seq_len=SEQ_LEN, tic_ids=train_tics)
    val_ds = FlarePhaseDataset(data, seq_len=SEQ_LEN, tic_ids=val_tics)
    test_ds = FlarePhaseDataset(data, seq_len=SEQ_LEN, tic_ids=test_tics)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMFlareClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #criterion = nn.CrossEntropyLoss()
    # Compute dynamic class weights
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    print(f"Starting training for {EPOCHS} epochs with {len(train_loader)} batches per epoch")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, 3), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete - Average Loss: {avg_loss:.4f}\n")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view(-1, 3), y.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"\n Model saved to {model_path}")

    all_labels = []
    for tic in train_tics:
        all_labels += data[tic]['flare_phase_labels']
    print(Counter(all_labels))

    return model


# ----------------------------
# Load Trained Model
# ----------------------------
def load_model(model_path=SAVE_PATH):
    """
    Loads a trained LSTM flare classifier model from disk.

    Parameters:
        model_path (str): Path to the saved model file.

    Returns:
        model (nn.Module): Loaded model in evaluation mode.
    """
    model = LSTMFlareClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model

# ----------------------------
# Evaluate Model
# ----------------------------
def evaluate_model(model, dataset_path=SYNTHETIC_PATH, split_path= SPLIT_PATH, visualize=True, save_report=True, output_dir="../outputs"):
    """
    Evaluates the model on the test dataset, prints classification metrics,
    optionally displays a confusion matrix, and saves evaluation results.

    Parameters:
        model (nn.Module): Trained model to evaluate.
        dataset_path (str): Path to the dataset pickle file.
        visualize (bool): Whether to display confusion matrix plot.
        save_report (bool): Whether to save accuracy and classification report to disk.
        output_dir (str): Folder path to save evaluation output files.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    with open(split_path, 'r') as f:
        splits = json.load(f)

    test_ds = FlarePhaseDataset(data, seq_len=SEQ_LEN, tic_ids=splits["test"])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().reshape(-1))
            all_labels.extend(y.cpu().numpy().reshape(-1))

    report_text = classification_report(all_labels, all_preds, digits=3)
    report = classification_report(all_labels, all_preds, digits=3, output_dict=True)
    acc = accuracy_score(all_labels, all_preds)

    print("Classification Report (Test Set):")
    print(report_text)
    print(f"Test Accuracy: {acc:.4f}")

    if save_report:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write("Classification Report (Test Set):\n")
            f.write(report_text)
            f.write(f"\nTest Accuracy: {acc:.4f}\n")
        print(f"Saved evaluation results to {output_dir}/metrics.txt")
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(report, f, indent=2)

    if visualize:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["None", "Rise", "Decay"])
        disp.plot(cmap="Blues")
        plt.title("Flare Phase Confusion Matrix")
        plt.tight_layout()

        # Save confusion matrix image
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved confusion matrix to {cm_path}")
        plt.show()


# ----------------------------
# Predict Single Sequence
# ----------------------------
def predict_single_sequence(model, flux_sequence):
    """
    Makes predictions on a single sequence of flux values.

    Parameters:
        model (nn.Module): Trained model.
        flux_sequence (np.ndarray): 1D numpy array of flux values.

    Returns:
        np.ndarray: Predicted phase labels for each timestep.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor(flux_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        output = model(seq_tensor)
        pred_labels = output.argmax(dim=-1).squeeze(0).cpu().numpy()
    return pred_labels
