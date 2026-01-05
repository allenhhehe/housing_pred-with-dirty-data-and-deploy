California Housing Price Prediction Project (with Dirty Data Handling and Deployment)
Project Overview
This is a complete end-to-end machine learning project using the classic California Housing dataset (built-in to scikit-learn). The main goals are:

Implement StandardScaler and gradient descent-based LinearRegression from scratch
Compare results with scikit-learn's official implementations
Prepare for handling "dirty data" scenarios (extensible foundation already in place)
Train and save the model
Deploy as an interactive web application using Flask, allowing users to input housing features in the browser and get instant price predictions (in USD)

Although the project name includes "dirty data", the current version works perfectly with clean data and provides a solid base for future experiments with missing values, outliers, or 
noise.
Project Structure


textsrc/
├── data.py               # Data loading
├── preprocess.py         # Train/val/test split, feature/target separation, custom StandardScaler
├── model_scratch.py      # Custom gradient descent Linear Regression
├── evaluate.py           # MAE and RMSE calculation
├── train.py              # Model training, comparison, saves model.pkl and scaler.pkl
├── app.py                # Flask web application (deployment entry point)
├── model.pkl             # Trained sklearn model (generated after running train.py)
├── scaler.pkl            # Custom scaler (generated after running train.py)
└── requirements.txt      # Dependencies (optional)
Quick Start
1. Environment Setup (Python 3.9+ recommended)
Bash# Clone the repository
git clone https://github.com/allenhhehe/housing_pred-with-dirty-data-and-deploy.git
cd housing_pred-with-dirty-data-and-deploy/src

# Install dependencies (virtual environment recommended)
pip install scikit-learn pandas numpy flask joblib
2. Train the Model
Bashpython train.py
This will print MAE/RMSE on validation and test sets (scratch and sklearn versions are nearly identical) and generate two files in the current directory:

model.pkl
scaler.pkl

3. Run the Web Application Locally
Bashpython app.py
Open your browser and visit: http://127.0.0.1:5000
Enter housing features and click "Predict House Price" to see the predicted price in USD!
4. Online Deployment (Recommended: Render - Free Tier Available)

Push the project to GitHub
Generate requirements.txt in the src directory:Bashpip freeze > requirements.txtThen manually add one line: gunicorn
Go to https://render.com → New → Web Service
Connect your GitHub repository
Settings:
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app

After deployment, you'll get a public URL where anyone can use your house price predictor!

Demo Preview

Medium income, average features → Predicted price around $200,000 ~ $300,000
High income, newer house, Bay Area location → Easily exceeds $500,000

Future Extensions

Add real dirty data handling (missing values, outliers, noise)
Feature engineering (e.g., clustering latitude/longitude, ocean proximity)
Support batch prediction via CSV upload
Try stronger models (RandomForest, XGBoost)

Author
Allen He (@allenhhehe)
Feel free to Star ⭐ and Fork! Contributions and improvements are welcome!
