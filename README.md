# ML Experiment Tracking & Reproducibility with Git, DVC & MLflow

A complete, production-ready Machine Learning project demonstrating data version control, experiment tracking, and pipeline reproducibility using **Git**, **DVC**, and **MLflow** on the Iris dataset.

---

## 📁 Project Structure

```text
ml dvc/
│
├── data/
│   ├── iris.csv               # Generated dataset (tracked by DVC)
│   └── iris.csv.dvc           # DVC tracking pointer for dataset
│
├── models/
│   └── model.pkl              # Saved Random Forest model artifact (joblib)
│
├── src/
│   └── train.py               # Model training & MLflow logging script
│
├── mlruns/                    # MLflow experiment tracking logs & artifacts
│
├── create_dataset.py          # Dataset generation script (scikit-learn Iris)
│
├── params.yaml                # Centralized hyperparameter configuration
│
├── dvc.yaml                   # DVC pipeline specification (stages, deps, outs)
│
├── dvc.lock                   # DVC lockfile registering stage hashes
│
├── requirements.txt           # Python dependencies list
│
├── README.md                  # Project documentation & terminal guide
│
└── .gitignore                 # Excluded directories (venv, cache, raw data)
```

---

## 🛠️ Step-by-Step Setup & Terminal Commands

### Step 1: Create Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Initialize Git Repository

```bash
git init
git add .gitignore requirements.txt params.yaml create_dataset.py src/train.py dvc.yaml README.md
git commit -m "Initial commit: ML project structure, scripts, and DVC config"
```

---

### Step 4: Initialize DVC (Data Version Control)

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

---

### Step 5: Generate and Track Dataset with DVC

Generate dataset locally:
```bash
python create_dataset.py
```

Track `data/iris.csv` with DVC:
```bash
dvc add data/iris.csv
git add data/iris.csv.dvc data/.gitignore
git commit -m "Track iris.csv dataset with DVC"
```

---

### Step 6: Run Single Training Script directly

```bash
python src/train.py
```

---

### Step 7: Run Reproducible DVC Pipeline

Execute all pipeline stages defined in `dvc.yaml`:
```bash
dvc repro
```

Commit pipeline lockfile to Git:
```bash
git add dvc.lock
git commit -m "Update DVC pipeline lockfile"
```

---

### Step 8: Hyperparameter Tuning & Experiment Comparison

Modify hyperparameters in `params.yaml`:
```yaml
train:
  n_estimators: 150
  max_depth: 5
  random_state: 42
  test_size: 0.2
```

Re-run the pipeline to automatically re-train and track new metrics:
```bash
dvc repro
```

---

### Step 9: Launch MLflow UI

View interactive metrics, parameters, and model artifacts:
```bash
mlflow ui
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

### Step 10: Push Project to GitHub

```bash
# Link your local repository to a remote GitHub repository
git remote add origin https://github.com/your-username/ml-dvc-mlflow-demo.git
git branch -M main
git push -u origin main
```

*(Optional) Configure DVC Remote Storage (e.g. AWS S3, Google Cloud Storage, or Local Drive):*
```bash
dvc remote add -d myremote /path/to/dvc/remote/storage
dvc push
```

---

## 🎯 Features & Best Practices Implemented

- **Data Versioning**: Large datasets tracked cleanly outside Git using `.dvc` tracking files.
- **Pipeline Reproducibility**: `dvc.yaml` defines input dependencies, code triggers, parameters, and generated outputs.
- **Experiment Tracking**: MLflow logs `n_estimators`, `max_depth`, `accuracy`, `precision`, `recall`, `f1_score`, and model artifacts.
- **Model Persistence**: Trained scikit-learn model serialized locally as `models/model.pkl` using `joblib`.
