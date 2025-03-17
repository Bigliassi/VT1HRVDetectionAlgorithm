# VT1HRVDetectionAlgorithm

Created by Dr. Marcelo Bigliassi (Florida International University)

# VT1 Synergy Detection (Interactive + Persistent Learning)

This repository contains two main scripts:

1. **`synergy_app.py`**  
   - A **Tkinter-based GUI** for loading a user’s data file (Excel/CSV), applying optional filtering, detecting synergy (partial synergy approach + optional global model), and allowing users to **select** a synergy region interactively.  
   - The script then **appends** synergy-labeled data to `global_labeled_data.csv` so that you can cumulatively build a synergy-labeled dataset for future training.

2. **`train_global_model.py`**  
   - A command-line script that **loads** all synergy-labeled samples from `global_labeled_data.csv`, trains a scikit-learn model (e.g., `LogisticRegression`, or any other classifier you choose), and saves the trained model as `global_model.pkl`.  
   - The synergy app, on startup, attempts to **load** this `global_model.pkl` and use it for a second synergy prediction line (the “global synergy pred”) in the synergy figure.

By combining these two scripts, you and your students can iteratively label synergy data from multiple files, re-train the global model, and watch the synergy detection **improve** over time.

---

## Features

- **Bulletproof Column Handling**:  
  - The app searches for synonyms of `Time`, `rMSSD`, `alpha1`, `HF`, and optionally `HR` in your data file.  
  - If any required column is not found (by any synonym), the app raises an error.

- **Robust Time Parsing**:  
  - If `Time` can be parsed as a date/time, we convert it to “seconds from the first row.”  
  - Otherwise, if it looks like `HH:MM:SS`, we parse that as a duration in seconds.  
  - If all else fails, we use the row index as time in seconds.

- **Filtering / Artifact Removal**:  
  - User can skip the first/last N seconds.  
  - Remove outliers in `rMSSD`/`HF` using a robust z-score approach (`artifact Z-threshold`).  
  - Optionally smooth `rMSSD`/`HF` with either a **Savitzky-Golay** filter or a **Moving Average**.

- **Fitness Logic** with HR**:  
  - The user picks “Low,” “Moderate,” or “High” fitness. The code sets a typical HR range (e.g. 125–150 for moderate).  
  - A user-defined tolerance is subtracted/added to that range.  
  - If an HR column is present, the synergy detection checks whether HR is within that range for awarding synergy points.

- **Partial Synergy** (Local)**:  
  - Weighted approach:  
    - 2 points for rMSSD slope < 0,  
    - 2 points for HR in the fitness-based range,  
    - 1 point for HF slope < 0,  
    - 1 point if alpha1 < 0.8.  
  - The code sums these and divides by the total possible (6 points) → synergy in [0..1]. If synergy >= 0.5, we’d call that synergy=1, but we only plot the fractional synergy line.

- **Global Model** (Trained on All Labeled Data)**:  
  - If you have previously run `train_global_model.py`, it produces a `global_model.pkl`.  
  - `synergy_app.py` attempts to load that model on startup.  
  - When you run synergy, it calls the model’s `predict(...)` for a second synergy line (the “Global synergy pred,” typically 0 or 1).

- **Interactive SpanSelector**:  
  - The synergy figure shows `rMSSD`, `HF`, `alpha1` on the main axis, synergy lines on a second axis, and HR on a third axis.  
  - The user can click/drag horizontally to define a synergy region `[xmin, xmax]` in seconds.  
  - Once defined, the “Save Region to Global Data” button is enabled, letting you label that region (1) and everything else (0) → appended to `global_labeled_data.csv`.

- **Persistent Learning**:  
  - Over time, multiple labeled synergy samples accumulate in `global_labeled_data.csv`.  
  - Run `train_global_model.py` to incorporate all that data into a new global model, saved to `global_model.pkl`.  
  - Next time you run the synergy app, it loads the updated model and presumably does better synergy detection for new files.

---

## Setup and Dependencies

- **Python 3.7+** recommended.  
- Libraries (install via conda or pip):  
  - `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `openpyxl` (if reading `.xlsx`)  
  - `joblib` (for saving/loading the model)  
- **Scripts**:  
  - `synergy_app.py`  
  - `train_global_model.py`  
  - Optionally a `global_labeled_data.csv` and `global_model.pkl` if you’ve already done some labeling/training.

Example install commands (for conda):
```bash
conda install -c conda-forge pandas numpy matplotlib scipy scikit-learn openpyxl
pip install joblib
```

---

## Usage

1. **Initial**:
   - Possibly no `global_labeled_data.csv` and no `global_model.pkl`.  
   - `synergy_app.py` will run, but it can’t do a “global synergy pred” line until you train a global model.

2. **`python synergy_app.py`**:
   - 1) **Load** a data file with “Load File (CSV/XLSX).”  
   - 2) (Optional) Adjust skip times, artifact threshold, filter method.  
   - 3) “Apply / Show Raw vs Filtered” if you want to see data changes.  
   - 4) “Run Synergy” → synergy figure appears with partial synergy line. If `global_model.pkl` is found, it also shows the “Global synergy pred.”  
   - 5) (Optional) Drag horizontally to define a synergy region `[xmin, xmax]`. Click “Save Region to Global Data.”  
     - This appends synergy-labeled rows to `global_labeled_data.csv`.  

3. **`python train_global_model.py`**:
   - Reads all synergy-labeled rows from `global_labeled_data.csv`.  
   - Trains e.g. a `LogisticRegression`, saves as `global_model.pkl`.  

4. **Improvement Over Time**:
   - As you and your students keep labeling synergy in different files, the CSV grows.  
   - Re-run `train_global_model.py` to incorporate all new labels.  
   - Next time you open `synergy_app.py`, it automatically loads the updated `global_model.pkl` and typically does better synergy detection for new data.

---

## Common Questions

- **My synergy lines are stuck at 0**  
  - Possibly your data doesn’t meet thresholds (like rMSSD slope, HR range, alpha1 < 0.8). Adjust thresholds or get more varied data.  
  - If your global model is always predicting 0 or 1, you might need more labeled data for it to generalize.

- **I see an error about missing columns**  
  - Check that your file has synonyms for `Time, rMSSD, alpha1, HF` (and optionally `HR`).  
  - The synonyms are in `synergy_app.py` → “synonyms” dict. Add more if needed.

- **I get an out-of-bounds time parse**  
  - If your `Time` column is unusual, the code tries HH:MM:SS or row index fallback. Check the logs or confirm your data format.

---

## Advanced Options

- **Adjust** partial synergy weighting. (In `partial_synergy_score()`, change `w_rMSSD`, `w_HR`, `w_HF`, `w_alpha`.)
- **Modify** thresholds for rMSSD slope (< -0.2?), HF slope (< -0.05?), alpha1 < 0.8, HR range, etc.
- **Switch** from logistic regression to random forest or any other model in `train_global_model.py`.
- **Add** a “test set” of synergy-labeled data to measure improvement over time or do cross-validation.

---

## Conclusion

This setup allows you to:

1. **Interactively define** synergy-labeled regions for partial synergy detection.  
2. **Persist** new synergy labels in a global CSV.  
3. **Retrain** a global model to gradually improve synergy detection across future data sets.

Enjoy your **VT1 synergy detection**!
