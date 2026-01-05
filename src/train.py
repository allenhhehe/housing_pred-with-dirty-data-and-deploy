import numpy as np
from sklearn.linear_model import LinearRegression
from data import load_data
from preprocess import split_xy,train_val_test_split,StandardScalerScrath
from model_scratch import linearRegressionGD
from evaluate import rmse, mae
import joblib
import os

def main():
    df=load_data()
    X,y=split_xy(df)
    X_train,y_train,X_val,y_val,X_test,y_test=train_val_test_split(X,y)

    scaler=StandardScalerScrath()
    X_train_s=scaler.fit_transform(X_train)
    X_val_s=scaler.transform(X_val)
    X_test_s=scaler.transform(X_test)

    scratch=linearRegressionGD(lr=0.05,epochs=3000).fit(X_train_s,y_train)
    pred_val_s=scratch.perdict(X_val_s)
    pred_test_s=scratch.perdict(X_test_s)

    sk=LinearRegression().fit(X_train_s,y_train)
    perd_val_k=sk.predict(X_val_s)
    perd_test_k=sk.predict(X_test_s)

    print("=== Validation ===")
    print("Scratch MAE/RMSE:",mae(pred_val_s,y_val),rmse(pred_val_s,y_val))
    print("Sklearn MAE/RMSE:",mae(perd_val_k,y_val),rmse(perd_val_k,y_val))

    print("\n=== Test ===")
    print("Scratch MAE/RMSE:",mae(pred_test_s,y_test),rmse(pred_test_s,y_test))
    print("Sklearn MAE/RMSE:",mae(perd_test_k,y_test),rmse(perd_test_k,y_test))

    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, 'model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')

    joblib.dump(sk, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("Model and scaler have been saved in the project directory!")

if __name__ == "__main__":
        main()
    

        

