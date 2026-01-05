import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class StandardScalerScrath:
    mean_: np.ndarray | None=None
    std_: np.ndarray  | None=None


    def fit(self,X:np.ndarray)->"StandardScalerScrath":
        self.mean_=X.mean(axis=0)
        self.std_=X.std(axis=0)
        self.std_[self.std_==0]=1.0 #to avoid division by zero
        return self
    
    def transform(self,X:np.ndarray)->np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted")
        return((X-self.mean_)/self.std_)
    
    def fit_transform(self,X:np.ndarray)->np.ndarray:
        return self.fit(X).transform(X)
    
def split_xy(df:pd.DataFrame,target_col:str="target"):
        X=df.drop(columns=[target_col]).to_numpy()
        y=df[target_col].to_numpy()
        return X,y
    
  
def train_val_test_split(X,y,seed=42):
        X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=seed)
        X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=seed)
        return X_train,y_train,X_val,y_val,X_test,y_test
    

    
    

    