# Predictive Maintenance — ML Workflow

This repository contains the interim Predictive Maintenance project, its dataset, notebook, and an automated GitHub Actions pipeline.

## Repository structure

```text
.
├── .github/workflows/pipeline.yml
├── data/engine_data.csv
├── notebooks/
│   ├── Akram_PredictiveMaintenance_Notebook.ipynb
│   └── Akram_PredictiveMaintenance_Notebook.html
├── src/
│   ├── validate_data.py
│   ├── prepare_data.py
│   └── train_model.py
├── artifacts/
├── requirements.txt
└── README.md
```

## What the workflow does

1. Validates the registered CSV schema, target values, and missing data.
2. Removes duplicate rows and creates a reproducible, stratified 80/20 train/test split.
3. Uploads `train.csv` and `test.csv` as a GitHub Actions artifact.
4. Downloads those split files in the next job.
5. Tunes Decision Tree, Random Forest, AdaBoost, and Gradient Boosting using stratified 3-fold cross-validation and weighted F1.
6. Selects the winning candidate using **training-only cross-validation**. The held-out test set is used only for final evaluation of the selected model and the dummy baseline.
7. Uploads the selected model, metadata, experiment log, CV comparison, and evaluation report as workflow artifacts.
8. On pushes to `main`, commits the generated model/evaluation files under `artifacts/model/`.

## Run locally

Use Python 3.11+ (Python 3.11 is used by the GitHub Actions workflow).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/validate_data.py
python src/prepare_data.py --input data/engine_data.csv --output-dir artifacts/splits
python src/train_model.py --split-dir artifacts/splits --output-dir artifacts/model
```

## Push to GitHub

1. Create a **new empty repository** on GitHub, for example `Akram-Predictive-Maintenance`.
2. Extract this project ZIP locally.
3. Open a terminal in the extracted folder and run:

```bash
git init
git add .
git commit -m "Initial predictive maintenance project"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

4. In GitHub, open **Actions** and select **Predictive Maintenance ML Pipeline**.
5. Check that both jobs pass and that the workflow artifacts are available.
6. If the workflow cannot push model outputs, confirm the repository's Actions setting allows workflows to write to the repository. Do not put personal access tokens or passwords in source files.

## Expected outputs

- `artifacts/splits/train.csv`
- `artifacts/splits/test.csv`
- `artifacts/model/best_predictive_maintenance_model.joblib`
- `artifacts/model/best_model_metadata.json`
- `artifacts/model/experiment_tracking.csv`
- `artifacts/model/cv_model_comparison.csv`
- `artifacts/model/model_evaluation.csv`
- `artifacts/model/classification_report.txt`

## Important notes

- The dataset's target label meanings must be confirmed before translating predictions into real maintenance actions.
- The GitHub Actions workflow is included as a template and must be pushed to your repository and run successfully before claiming the workflow has been verified.
- The interim business report PDF is a separate submission deliverable and is not included in this repository ZIP.
