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

    This dataset:
    - Applies Savitzky-Golay smoothing to each light curve to isolate the flare-only signal
    - Segments each light curve into overlapping sequences
    - Optionally filters out sequences that contain no flare activity
    - Maintains original flare phase labels: 0 = no flare, 1 = rise, 2 = decay

    Args:
        lightcurve_dict (dict): Dictionary of TIC IDs and their flare-injected light curve data.
        seq_len (int): Length of each input sequence.
        tic_ids (list): List of TIC IDs to include in the dataset.
        stride (int): Step size between overlapping sequences (default: STRIDE).
    Returns:
        torch.utils.data.Dataset: Each item is a tuple of:
            - flare-only flux sequence (shape: [seq_len, 1])
            - phase label sequence (shape: [seq_len])
    """
    def __init__(self, lightcurve_dict, seq_len, tic_ids, stride=STRIDE, include_nonflare_ratio=0.05):
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
                # insist that we have some non-flaring sequences and some sequences where both rise and decay are present in the sequence so the model is able to learn and avoid class imbalance
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
        random.shuffle(self.samples) # randomly reorder list in place so DataLoader is not grabbing sequences grouped in a bias way

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
    LSTM-based model with attention mechanism to classify each timestep
    in a light curve sequence into one of the flare phases.

    Parameters:
        input_size (int): Size of the input feature vector (default: 1).
        hidden_size (int): Number of LSTM hidden units.
        num_layers (int): Number of LSTM layers.
        num_classes (int): Number of flare phase classes (default: 3).
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
        alpha: weighting factor for classes (tensor of shape [num_classes])
        gamma: focusing parameter for modulating factor (1-p)
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

def train_model(dataset_path=SYNTHETIC_PATH, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LR, seq_len=SEQ_LEN, use_attention= True, input_size=1, hidden_size= 128, num_layers=3, num_classes=3, dropout=0.3, split_path= SPLIT_PATH, model_path= SAVE_PATH):
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

    train_ds = FlarePhaseDataset(data, seq_len, tic_ids=train_tics)
    val_ds = FlarePhaseDataset(data, seq_len, tic_ids=val_tics)
    test_ds = FlarePhaseDataset(data, seq_len, tic_ids=test_tics)

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
    Loads a trained LSTM flare classifier model from disk.

    Parameters:
        model_path (str): Path to the saved model file.

    Returns:
        model (nn.Module): Loaded model in evaluation mode.
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
def evaluate_model(model,model_path= None,batch_size=BATCH_SIZE, seq_len=SEQ_LEN, dataset_path=SYNTHETIC_PATH, split_path= SPLIT_PATH, visualize=True, save_report=True, output_dir="../outputs"):
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

    test_ds = FlarePhaseDataset(data, seq_len, tic_ids=splits["test"])
    test_loader = DataLoader(test_ds, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(model_path=model_path)
    
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
