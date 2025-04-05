"""
lstm_flare_classifier.py

This module provides functionality for training, evaluating, and using an LSTM-based neural network
(with optional attention mechanism) to classify each timestep in a light curve as one of three 
flare phases: 0 (no flare), 1 (rise), 2 (decay). It includes:

- A PyTorch Dataset for preprocessed flare light curves
- LSTM classifier model with optional attention and dropout
- Focal loss function for handling class imbalance
- Utilities to compute dynamic class weights
- Training loop with early stopping and model checkpointing
- Model saving/loading
- Evaluation with confusion matrix and classification report

Expected input is a dictionary of synthetic light curves injected with flares,
containing keys: 'time', 'synthetic_flux', and 'flare_phase_labels'.

Example usage:
    from lstm_flare_classifier import train_model, evaluate_model, load_model
    model = train_model(dataset_path='data/synthetic_lightcurves.pkl')
    evaluate_model(model)
    loaded_model = load_model()
"""

import os
import json
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.signal import savgol_filter

# ----------------------------
# Config
# ----------------------------
SEQ_LEN = 200
STRIDE= SEQ_LEN/2 #50% overlap is common in signal processing
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
SAVE_PATH = "../models/lstm_flare_classifier.pt"
SPLIT_PATH = "../models/splits.json"
SYNTHETIC_PATH= "../data/synthetic_lightcurves.pkl"

# ----------------------------
# Dataset Class (Star-based Splits)
# ----------------------------
class FlarePhaseDataset(Dataset):
    """
    PyTorch Dataset for flare phase classification from synthetic TESS light curves.

    Each sample is a sequence of flare-only flux (smoothed baseline subtracted), 
    along with the corresponding phase labels.

    - Applies Savitzky-Golay smoothing to remove the stellar baseline.
    - Extracts overlapping sequences from each TIC light curve.
    - Includes a balance of flaring (rise+decay) and optionally non-flaring segments.

    Args:
    lightcurve_dict (dict): Dictionary mapping TIC IDs to flare-injected light curve data.
    tic_ids (list): List of TIC IDs to include.
    seq_len (int): Number of timesteps in each extracted sequence.
    stride (int): Step size for windowed sequence extraction.
    include_nonflare_ratio (float): Ratio of non-flaring sequences to inject to improve class balance.
    Returns:
        torch.utils.data.Dataset: Each item is a tuple of:
            - flare-only flux sequence (shape: [seq_len, 1])
            - phase label sequence (shape: [seq_len])
    """
    def __init__(self, lightcurve_dict, tic_ids, seq_len= SEQ_LEN, stride=STRIDE, include_nonflare_ratio=0.05):
        self.samples = []
        nonflare_candidates= []
        for tic in tic_ids:
            entry = lightcurve_dict[tic]
            raw_flux = entry['synthetic_flux']
            labels = entry['flare_phase_labels']

            if len(raw_flux) < 101:
                continue  # Not enough points for smoothing

            # Applying smoothing to signal
            baseline = savgol_filter(raw_flux, window_length=101, polyorder=3, mode='interp')
            flare_flux = raw_flux - baseline

            #slide over light curve 
            for i in range(0, len(flare_flux) - seq_len, int(stride)):
                label_seq = labels[i:i+seq_len]
                flux_seq = flare_flux[i:i+seq_len]
                label_counts = Counter(label_seq)
                # insist that we have some non-flaring sequences and some sequences where both rise 
                #and decay are present in the sequence so the model is able to learn and avoid class imbalance
                if 1 in label_seq and 2 in label_seq:
                    self.samples.append((flux_seq, label_seq))
                elif label_counts[0] > (label_counts[1] + label_counts[2]):
                #elif set(label_seq) == {0}:
                    nonflare_candidates.append((flux_seq, label_seq))
        # Add non-flaring sequences based on ratio
        num_flaring = len(self.samples)
        desired_nonflare = int((include_nonflare_ratio / (1 - include_nonflare_ratio)) * num_flaring)
        if desired_nonflare > 0:
            selected_nonflare = random.sample(nonflare_candidates, 
                                              min(desired_nonflare, 
                                              len(nonflare_candidates)))
            self.samples.extend(selected_nonflare)
        # randomly reorder list in place so DataLoader is not grabbing sequences grouped in a bias way
        random.shuffle(self.samples) 

        # Count total points per class (flattened across all sequences)
        all_labels = []
        for _, label_seq in self.samples:
            all_labels.extend(label_seq)
        label_counts = Counter(all_labels)
        print("Total label counts (FULL DATASET)::", label_counts)


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
    LSTM-based classifier for flare phase prediction at each timestep in a sequence.
    
    Args:
        input_size (int): Number of input features (default: 1 for 1D flux).
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
        num_classes (int): Number of output classes (default: 3).
        dropout (float): Dropout probability between LSTM and final layer.
        use_attention (bool): If True, applies attention weighting to LSTM outputs.
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, num_classes=3, dropout=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        if use_attention:
            self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout= nn.Dropout(dropout) # dropout to prevent overfitting

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
            lstm_out = lstm_out * attn_weights  # elementwise multiplication
        out=self.dropout(lstm_out)        
        out = self.fc(out)
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal loss for addressing class imbalance by focusing learning on hard-to-classify examples.
        
        Args:
            alpha (tensor): Optional class weights.
            gamma (float): Focusing parameter. Higher values increase emphasis on difficult examples.
            reduction (str): Aggregation method ('mean', 'sum', or 'none') for batch loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] (logits)
        targets: [batch_size] (labels)
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ----------------------------
# Train Model
# ----------------------------

# Compute class weights dynamically
def compute_class_weights(dataset):
    """
    Computes inverse-frequency class weights from a labeled dataset.
    
    Args:
        dataset (Dataset): Instance of FlarePhaseDataset or equivalent.
    
    Returns:
        torch.Tensor: Weight for each class (shape: [num_classes]).
    """
    from collections import Counter
    label_counts = Counter()
    for _, label_seq in dataset:
        label_counts.update(label_seq.numpy().tolist())

    total = sum(label_counts.values())
    weights = [total / label_counts.get(i, 1) for i in range(3)]
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Computed class weights: {weights}")
    return weights

def train_model(dataset_path=SYNTHETIC_PATH, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LR, seq_len=SEQ_LEN, use_attention= True, input_size=1, hidden_size= 128, num_layers=3, num_classes=3, dropout=0.3, stride= STRIDE, split_path= SPLIT_PATH, model_path= SAVE_PATH):
    """
    Train an LSTM flare classifier on the flare-injected dataset using star-based train/val/test splits.
    
    - Uses dynamic class weights.
    - Supports optional attention in the model.
    - Includes early stopping based on validation loss.
    - Saves model and config to disk.
    
    Args:
        dataset_path (str): Path to pickled synthetic lightcurve dictionary.
        model_path (str): Where to save the model.
        split_path (str): Where to save the train/val/test split.
        batch_size (int): Number of samples per batch.
        epochs (int): Total number of training epochs.
        learning_rate (float): Optimizer learning rate.
        seq_len (int): Length of input sequences.
        use_attention (bool): Whether to apply attention in the model.
        input_size (int): Feature size of each timestep.
        hidden_size (int): Hidden dimension of LSTM.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        num_classes (int): Number of phase classes.
    
    Returns:
        nn.Module: Trained model.
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

    train_ds = FlarePhaseDataset(data, tic_ids=train_tics, seq_len= seq_len, stride=stride)
    val_ds = FlarePhaseDataset(data, tic_ids=val_tics, seq_len= seq_len, stride= stride)
    test_ds = FlarePhaseDataset(data, tic_ids=test_tics, seq_len= seq_len, stride= stride)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    
    model = LSTMFlareClassifier(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_classes=num_classes,
                                dropout= dropout,
                                use_attention=use_attention)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #criterion = nn.CrossEntropyLoss()
    # Compute dynamic class weights
    class_weights = compute_class_weights(train_ds).to(device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    #criterion = FocalLoss(gamma=2)
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    best_val_loss= float('inf')
    epochs_no_improve= 0
    patience=3 
    
    print(f"Starting training for {epochs} epochs with {len(train_loader)} batches per epoch")
    for epoch in range(epochs):
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
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Early stopping to prevent model from continuing past optimal val performance (overfitting)
        if avg_val_loss < best_val_loss:
            best_val_loss= avg_val_loss
            epochs_no_improve=0
            #save the best model
            # Save model state 
            torch.save(model.state_dict(), model_path)
            print(f"\n Model saved to {model_path}")
        
            # Save model configuration as JSON
            model_config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "use_attention": use_attention,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "seq_len": seq_len,
            "stride": stride,
            "model_path": model_path}
            
            config_path = model_path.replace(".pt", "_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"Model config saved to {config_path}")
        else:
            epochs_no_improve +=1
            print(f"No improvement for {epochs_no_improve} epcoh(s)")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    

    all_labels = []
    for _, label_seq in train_ds:
        all_labels.extend(label_seq.numpy().tolist())
    print("Training set label distribution:", Counter(all_labels))

    return model


# ----------------------------
# Load Trained Model
# ----------------------------
def load_model(model_path=SAVE_PATH):
    """
    Loads a trained LSTM model from disk using its saved configuration.
    
    Args:
        model_path (str): Path to .pt model file.
    
    Returns:
        nn.Module: The loaded model in eval mode.
    """
    config_path = model_path.replace(".pt", "_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = LSTMFlareClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout= config["dropout"],
        use_attention=config["use_attention"])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model

# ----------------------------
# Evaluate Model
# ----------------------------
def evaluate_model(model,model_path= None,batch_size=BATCH_SIZE, seq_len=SEQ_LEN, stride= STRIDE, dataset_path=SYNTHETIC_PATH, split_path= SPLIT_PATH, visualize=True, save_report=True, output_dir="../outputs"):
    """
    Evaluate a trained flare classifier on the test set using the saved split file.
    
    - Prints classification report.
    - Outputs confusion matrix.
    - Saves evaluation metrics to disk.
    
    Args:
        model (nn.Module): A trained model (optional if loading from disk).
        model_path (str): Path to model checkpoint to load (optional if model is provided).
        batch_size (int): Evaluation batch size.
        seq_len (int): Sequence length used for slicing.
        stride (int): Stride for generating sequences.
        dataset_path (str): Path to synthetic lightcurve data.
        split_path (str): JSON file specifying train/val/test splits.
        visualize (bool): Whether to display confusion matrix.
        save_report (bool): Whether to save metrics to disk.
        output_dir (str): Where to store results.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    with open(split_path, 'r') as f:
        splits = json.load(f)

    test_ds = FlarePhaseDataset(data, tic_ids=splits["test"], seq_len= seq_len, stride= stride)
    test_loader = DataLoader(test_ds, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(model_path=model_path)
    
    model.to(device)
    model.eval() # ensures eval model for both in memory and pre loaded models

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=-1)
            # print firsts equence only for sanity check
            if i==0:
                print("First batch sanity check")
                print("Preds:", preds[0].cpu().numpy())
                print("True: ", y[0].cpu().numpy())
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
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["None", "Rise", "Decay"])
        disp.plot(cmap="Blues")
        plt.title("Flare Phase Confusion Matrix")
        plt.tight_layout()

        # Save confusion matrix image
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved confusion matrix to {cm_path}")
        plt.show()

