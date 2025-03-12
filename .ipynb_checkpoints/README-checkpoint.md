# STA2543_Stellar_Flares

## Project Overview

This project focuses on detecting and predicting stellar flares—sudden bursts of radiation caused by magnetic reconnection on stars. The goal is to analyze time-series light curve data from TESS (Transiting Exoplanet Survey Satellite), generate synthetic flare data to expand the dataset, and train an LSTM-based deep learning model to predict flares before they occur.

## Motivation

Stellar flares play a crucial role in understanding space weather and the habitability of exoplanets. By leveraging machine learning, we aim to:

- Identify **pre-flare signatures** in light curves.
- Improve **flare detection** by augmenting real observations with synthetic data.
- Develop an **LSTM-based model** to predict flares before they happen.

## Project Structure

This project consists of three key components:

### 1. Exploratory Data Analysis (EDA)

**File:** `EDA.ipynb`

- Jupyter Lab was used to run all analyses and visualizations.
- Performs initial analysis of light curves from TESS.
- Detects stellar flares using the **Altaipony package**.
- Analyzes flare frequency, intensity, and periodicity across **M-dwarfs, K-dwarfs, G-type, and F-type stars**.
- Includes visualizations such as **flare count distributions, variability analysis, and clustering.**

### 2. Generating Synthetic Data

**File:** `generate_synthetic.ipynb`

- Jupyter Lab was used to run this notebook.
- Expands the dataset by **injecting synthetic flares** into light curves.
- Ensures flares follow realistic astrophysical properties (sharp rise, exponential decay).
- Labels **pre-flare states** to assist with machine learning predictions.
- Stores the augmented dataset in `synthetic_lightcurves.pkl` for training.

### 3. LSTM-Based Flare Prediction

**File:** `LSTM_pipeline.ipynb`

- Jupyter Lab was used to run this notebook.
- Loads the **synthetic dataset** and preprocesses it for training.
- Uses **LSTM neural networks** to model time-series flux variations.
- Predicts **whether a flare will occur in the next `N` timesteps**.
- Implements **data normalization, loss balancing, and evaluation metrics** to improve prediction accuracy.

## Installation & Requirements

Ensure you have the required dependencies installed: WILL FILL OUT LATER FULLY WITH .ENV FILE

```bash
pip install numpy pandas matplotlib seaborn tensorflow astropy altaipony
```

## How to Run the Project

1. Clone this repo, cd to directory and activate enviornment (will provide enviornment file and instructions later)
2. Open jupyter lab in the dirctory
    ```bash
   jupyter lab
   ```
4. **Run all cells in** `EDA.ipynb` to explore the dataset and visualize flares in **Jupyter Lab** and create `processed_lightcurves.pkl`.
5. **Run all cells in** `generate_synthetic_data.ipynb` to create `synthetic_lightcurves.pkl`.
6. **Train the LSTM model** by running opening `LSTM_pipeline.ipynb`. Run all cells to train the model and evaluate flare prediction accuracy.

## Future Improvements

- Implement **attention-based LSTMs** to focus on key pre-flare signals.
- Test out deeper LSTM architectures and perform hyper-parameter tuning
- Experiment with **GANs or Variational Autoencoders (VAEs)** for more realistic flare augmentation.

## Contributors

**Naomi Kothiyal** – University of Toronto

---

**Goal:** Advance flare prediction research using machine learning for astrophysical applications.

