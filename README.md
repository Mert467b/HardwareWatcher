# CPU Throttle Predictor

A machine learning pipeline that predicts whether a laptop's CPU will go through a thermal throttle in the next 30 seconds. Uses from scratch implementation of Decision Tree and random forest. Has library implemented XGBoost for comparison.

---

## What It Does

This project logs real time hardware telemetry (CPU/GPU utilization, temperature etc.) and trains a classifier to predict throttling events.

---

## Project Structure

```
HardwareWatcher/
├── data_fix/               # Notebooks for data maintenance
│   ├── data_fix.ipynb      # Repairs session IDs and timestamps
│   └── data_process.ipynb  # Initial data cleaning and regex formatting
├── get_data/               
│   └── get_data.py         # Real-time hardware telemetry logger
├── src/                    
│   ├── __init__.py         # Makes the directory a Python package
│   ├── models.py           # Custom Random Forest & Decision Tree classes
│   ├── utils.py            # Preprocessing, metrics, and feature engineering
│   └── visualization.py    # Confusion matrix and validation plotting
├── telemetry/              
│   └── telemetry.py        
├── .env.example            # Template for your local environment variables
├── .gitignore              # Files and folders for Git to ignore
├── main.py                 # CLI entry point to run the ML pipeline
└── requirements.txt        # List of Python dependencies for the project
└── streamlit_app.py        # Web-based dashboard for predictions and analysis
```

---
## Models
Two classifiers are available, toggled via 'TASK' in 'main.py':

**'Random Forest'**:
- Custom implementation built from scratch using only NumPy
- Gini impurity splitting with percentile based-threshold search
- Class weight balancing
- Configurable nature: allows to change the values of tree count, depth and leaf size

**'XGBoost'**:
- Ready to use model. Called from library. For easy comparison.

---

## Features used:
The model is trained on a mix of current state, rolling statistics, and lag features:

| Category | Features |
|---|---|
| Categorical | Activity type, surface type, CPU boost mode |
| Current state | CPU/GPU temp, power draw, utilization |
| Rolling stats | 30s and 60s rolling mean/std for temp and power |
| Lag features | CPU/GPU temp, power, utilization at t-5, t-10 |
| Derived | CPU/GPU temperature slope, frequency headroom |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect data

Set up your `.env` file with your [LibreHardwareMonitor](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor) server credentials:

```
MY_URL=http://localhost:8085/data.json
MY_USER=your_username
MY_PASS=your_password
```

Then log hardware telemetry (records at ~1 Hz):

```bash
python get_data.py
```

Set `CURRENT_ACTIVITY`, `SURFACE_TYPE`, and `CPU_BOOST_TYPE` at the top of the file to label your sessions.

### 3. (Optional) Clean your data

If your CSV has formatting artifacts or mixed sessions, run the two notebooks in order:

```
data_process.ipynb   # Strips unit strings, fixes typos
data_fix.ipynb       # Reconstructs session IDs and slopes
```

### 4. Train and evaluate

```bash
python main.py
```

Configure at the top of `main.py`:

```python
TASK = 'RandomForest'   # or 'XGBoost'
RUN_PLOTS = True        # Generates confusion matrix + validation plots
DATA_FILE = 'data_logs.csv'
```

### 5. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard is a full interactive UI for the pipeline. Upload your CSV, tune hyperparameters, and explore results across four tabs:

| Tab | What it shows |
|---|---|
|**Data Explorer**| Row count, missing %, session count, column distributions |
|**Dataset Config**| Auto mode (CPU throttle CSV) or Custom mode (any tabular CSV with manual feature/target selection) |
|**Model Results**| Accuracy, precision, recall, F1, train/test gap indicator, confusion matrix, color-coded classification report, feature importance chart |
|**Validation Tests**| Temporal degradation curve and walk-forward validation with variance verdict |

Hyperparameters (tree count, depth, leaf size, decision threshold) are all tunable from the sidebar without touching any code.

---

## Evaluation

The pipeline reports accuracy, precision, recall, F1, and a train/test gap check. Two validation plots are generated automatically:

- **Temporal Degradation** — does accuracy improve as more training data is added?
- **Walk-Forward Validation** — is performance stable across different time splits? Accuracy std < 0.02 is flagged as consistent, > 0.05 as high variance.

---

## Requirements

- Windows (LibreHardwareMonitor runs on Windows)
- NVIDIA GPU (for GPU telemetry via `pynvml`)
- Python 3.10+

## Notes
Used AI assistance for Streamlit dashboard implementation (`streamlit_app.py`)
