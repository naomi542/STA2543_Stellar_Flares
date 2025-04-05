# STA2543 Stellar Flares Phase Classification

## Project Overview

This project performs **flare phase classification** (None, Rise, Decay) on synthetic stellar light curves. It uses an LSTM-based neural network with optional attention to predict the phase of each timestep in a flux sequence. All steps are orchestrated from a single pipeline notebook: `pipeline.ipynb`. I analyze time-series light curve data from TESS (Transiting Exoplanet Survey Satellite) and generate synthetic flare data to expand the dataset.
---

## Motivation

Stellar flares play a crucial role in understanding space weather and the habitability of exoplanets. By leveraging machine learning, I aim to:

- Identify **pre-flare signatures** in light curves.
- Improve **flare detection** by augmenting real observations with synthetic data.
- Develop an **LSTM-based model** to predict flares before they happen. This is the long-term goal. With this project, I aim to prove that a predictive model is able to distinguish the different phases of a stellar profile. With more advanced modelling and larger dataset, we should be able to predict flares as the lightcurve is fed into an LSTM network in sequences.
---


## Project Structure
```
StellarFlares/
│
├── data/                         # Input and synthetic datasets
├── models/                       # Trained model checkpoints
├── outputs/                      # Evaluation metrics and confusion matrices
├── scripts/                      # All supporting Python scripts
│   ├── inject_synthetic_flares.py
│   ├── lstm_flare_classifier.py
│   ├── load_data.py
│   ├── plot_utils.py
│   ├── summarize.py
│   └── star_catalog.py
│
├── pipeline.ipynb                # Main training + evaluation notebook. Fully reproducible guide notebook/vignette
├── requirements.txt              # Python dependencies
└── README.md
```
---
## Main Features
- Synthetic flare generation using Kepler-inspired flare profiles
- LSTM-based model with optional attention
- Configurable hyperparameters via notebook
- Balanced sampling and class weighting
- Detailed evaluation reports and confusion matrices

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/naomi542/STA2543_Stellar_Flares.git
cd STA2543_Stellar_Flares
```

### 2. Set up the Python environment (recommended via conda)
```bash
conda create -n StellarFlares python=3.10 -y
conda activate StellarFlares
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter lab
```

Then open `pipeline.ipynb` and run all cells to execute the full pipeline:  
- load data  
- inject flares  
- train LSTM  
- evaluate model  
- save outputs (metrics + confusion matrix)

---

## Model Summary

- **Architecture**: LSTM with optional Attention
- **Loss Function**: Weighted Cross Entropy
- **Input**: Flux time series of a single star
- **Output**: Predicted flare phase at each timestep

---

## Experiments

The notebook allows sweeping through hyperparameters:
- Sequence length
- Batch size
- Dropout rate
- Number of layers
- Use of attention
- Hidden size
- Stride and overlap

See the **Experiments Table** inside `pipeline.ipynb` for full details.

## Data Flow: From Inputs to Outputs

This project processes TESS light curves to identify stellar flare phases through a synthetic flare injection pipeline and LSTM-based classification model. Below is a detailed breakdown of how data flows through the codebase:

### Input Data

- The file `star_catalog.py` defines the specific stars we analyze, listed by their TIC IDs.
- These IDs are passed into the function `process_star_sample()` in `load_data.py`, which:
  - Downloads and detrends the raw TESS light curves.
  - Saves the processed light curves as a pickle file to:

    ```
    ../data/processed_lightcurves.pkl
    ```

- The function `plot_raw_lightcurves()` in `plot_utils.py` visualizes these raw light curves and stores plots in:

    ```
    ../data/figures/raw/
    ```

### Synthetic Flare Generation

- The raw pickle from `processed_lightcurves.pkl` is read into the function `inject_into_lightcurves()` in `inject_synthetic_flares.py`.
- Synthetic flares are generated and injected based on Kepler-inspired templates and astrophysical morphology.
- The resulting flare-augmented light curves are saved to:

    ```
    ../data/synthetic_lightcurves.pkl
    ```

- Corresponding plots of flare-injected light curves are saved to:

    ```
    ../data/figures/synthetic/
    ```

### LSTM Model Training and Evaluation

- The LSTM model in `lstm_flare_classifier.py` reads from:

    ```
    ../data/synthetic_lightcurves.pkl
    ```

- During training, model artifacts are saved to the `models/` directory, including:
  - Model checkpoints:
    ```
    ../models/lstm_flare_classifier_<run_name>.pt
    ```
  - Configuration metadata:
    ```
    ../models/lstm_flare_classifier_<run_name>_config.json
    ```
  - Dataset splits for reproducibility:
    ```
    ../models/splits_<run_name>_config.json
    ```

- Evaluation results and summary plots are stored under:

    ```
    ../outputs/<run_name>/
    ```

This modular structure ensures that raw inputs, intermediate artifacts, and final outputs are all tracked and accessible at every stage of the pipeline.


## Outputs and Logging

All generated outputs and intermediate artifacts are automatically saved to dedicated directories during runtime. Here's how they are organized:

### `data/`
- Contains both original and synthetic datasets.
- Pickled files such as `processed_lightcurves.pkl` and `synthetic_lightcurves.pkl` are saved here after executing the respective data preparation and synthetic flare injection steps in `pipeline.ipynb`.

### `models/`
- Stores model checkpoints after each training run.
- Filenames follow the pattern:  
  `lstm_flare_classifier_<run_name>.pt`  
  Configuration metadata for the model is saved as:  
  `lstm_flare_classifier_<run_name>_config.json`
  Train, Test and Validation splits for the run are saved as:
  `splits_<run_name>_config.json`

### `outputs/`
- Contains evaluation metrics and confusion matrices for each run.
- Files saved here include:
  - `metrics.txt`: Human-readable classification report and test accuracy.
  - `metrics.json`: Structured classification report with precision, recall, F1-score for each class.
  - `confusion_matrix.png`: Visual confusion matrix for the test set.
- Subfolders are automatically created per run, based on the `run` name defined in the notebook.

### Logging During Training
- `pipeline.ipynb` prints training and validation loss after each epoch.
- If the validation loss improves, the model is saved to disk automatically.
- Early stopping is applied with a patience of 3 if no improvement is seen for a number of epochs.

This structure ensures full reproducibility and simplifies comparison across different model configurations.


##  Requirements

All dependencies are listed in `requirements.txt`:
- `torch`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `scipy`
- `tqdm`
- `jupyterlab`
- `altaipony`
- `sklearn`

---

## Contributors

**Naomi Kothiyal** 
MScAC Program, University of Toronto  
`naomi.kothiyal@mail.utoronto.ca`

---

**Goal:** Advance flare prediction research using machine learning for astrophysical applications.

