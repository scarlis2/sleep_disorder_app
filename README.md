# Sleep Disorder Screening App

## Project Overview
This Streamlit application supports the capstone project:
Wearable Data and Machine Learning for Early Detection of Sleep Disorders.

The app loads a trained Random Forest model and allows the user to upload a CSV
file of wearable-inspired physiological features to generate sleep stage predictions.

## Files in This Folder
- `app.py` - main Streamlit application
- `rf_model.pkl` - trained Random Forest model file
- `sample_input.csv` - sample input file for testing
- `requirements.txt` - Python packages needed to run the app
- `README.md` - instructions for setup and use

## How to Run the App
1. Open Anaconda Navigator.
2. Open the terminal for your `sleepapp` environment.
3. Navigate to this project folder.
4. Run:

```bash
streamlit run app.py
```

## How to Use the App
1. Launch the app in your browser.
2. Click the upload button.
3. Select `sample_input.csv` or another CSV file with matching feature columns.
4. Review:
   - uploaded data preview
   - predicted sleep stages
   - prediction distribution chart
   - prediction probabilities (if available)

## Important Notes
- The uploaded CSV must match the same feature structure used during model training.
- If the feature names do not match, the model will return an error.
- This app is intended for academic demonstration and early screening support only.

## How to Create rf_model.pkl
In your Jupyter notebook, after training the Random Forest model, run:

```python
import joblib
joblib.dump(rf, "rf_model.pkl")
```

Then move `rf_model.pkl` into this project folder.

## How to Create sample_input.csv
In your Jupyter notebook, run:

```python
X_test.head(20).to_csv("sample_input.csv", index=False)
```

Then move `sample_input.csv` into this project folder.
