# California Housing Price Prediction Project (with Dirty Data Handling and Deployment)

## Project Overview

This is a complete end-to-end machine learning project using the classic **California Housing dataset** (built-in to scikit-learn). The main goals are:

- Implement `StandardScaler` and gradient descent-based `LinearRegression` from scratch
- Compare results with scikit-learn's official implementations
- Prepare for handling "dirty data" scenarios (extensible foundation already in place)
- Train and save the model
- Deploy as an interactive web application using **Flask**, allowing users to input housing features in the browser and get instant price predictions (in USD)

Although the project name includes "dirty data", the current version works perfectly with clean data and provides a solid base for future experiments with missing values, outliers, or noise.

## Project Structure
.
├── README.md
├── requirements.txt          # Project dependencies
└── src/
├── data.py               # Data loading
├── preprocess.py         # Train/val/test split, feature/target separation, custom StandardScaler
├── model_scratch.py      # Custom gradient descent Linear Regression
├── evaluate.py           # MAE and RMSE calculation
├── train.py              # Model training, comparison, saves model.pkl and scaler.pkl
├── app.py                # Flask web application (deployment entry point)
├── dirty_experiment.py   # Dirty data experiments (noise, missing values, etc.)
└── plots.py              # Visualization helpers
text> **Note**: `model.pkl` and `scaler.pkl` are generated in the `src/` directory after running `train.py` and are **not** committed to the repository.

## Quick Start

### 1. Environment Setup (Python 3.9+ recommended)

```bash
# Clone the repository
git clone https://github.com/allenhhehe/housing_pred-with-dirty-data-and-deploy.git
cd housing_pred-with-dirty-data-and-deploy

# Install dependencies (one command!)
pip install -r requirements.txt && python -m src.train
2. Train the Model
Bashpython -m src.train
This will:

Print MAE/RMSE on validation and test sets (scratch and sklearn versions are nearly identical)
Generate model.pkl and scaler.pkl in the src/ directory

3. Run the Web Application Locally
Bashpython -m src.app
Open your browser and visit: http://127.0.0.1:5000
Enter housing features and click "立即预测房价" to see the predicted price in USD!
4. Online Deployment (Recommended: Render - Free Tier Available)

Your project is already on GitHub and ready for deployment!
Go to https://render.com → Dashboard → New → Web Service
Connect your GitHub account and select this repository
Configure the following settings:
Build Command: pip install -r requirements.txt
Start Command: gunicorn src.app:app

Click Create Web Service
Wait 2-5 minutes for deployment to complete
You will receive a public URL (e.g., https://your-app.onrender.com) where anyone can use your house price predictor!

Tip: On the free tier, the app may take 10-30 seconds to wake up on the first visit. Subsequent visits will be instant.
Demo Preview

Medium income, average features → Predicted price around $200,000 ~ $300,000
High income, newer house, Bay Area location → Easily exceeds $500,000

Future Extensions

Add real dirty data handling (missing values, outliers, noise injection)
Feature engineering (e.g., clustering latitude/longitude, ocean proximity)
Support batch prediction via CSV upload
Try stronger models (RandomForest, XGBoost)

Author
Allen He (@allenhhehe)
Feel free to Star ⭐ and Fork! Contributions and improvements are welcome!
