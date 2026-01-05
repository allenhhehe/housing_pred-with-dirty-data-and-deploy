import numpy as np
import pandas as pd
from data import load_data
def make_dirty(df:pd.DataFrame,
               nan_cols,
               outlier_cols,
               scale_cols,
               nan_ratio=0.1,
               outlier_factor=10.0,
               scale_factor=1000.0,
               seed=42):
    rng=np.random.default_rng(seed)
    df=df.copy()
    n=len(df)
    for col,ratio in nan_cols.items(): # e.g. {"MedInc":0.05, "AveRooms":0.10} set some Null figure
        idx=rng.choice(n,size=int(n*ratio),replace=False)
        df.iloc[idx,df.columns.get_loc(col)]=np.nan
    
    for col in outlier_cols:  # e.g. ["AveOccup"]
        idx=rng.choice(n,size=max(1,int(n*0.01)),replace=False) #1% strang figure
        df.iloc[idx,df.columns.get_loc(col)]*=outlier_factor

    for col in scale_cols: # e.g. ["Population"]
        df[col]=df[col]*scale_factor

    return df
    

def clip_outliers(df:pd.DataFrame,col,q_low=0.01,q_high=0.99):
    df=df.copy()
    lo,hi=df[col].quantile(q_low),df[col].quantile(q_high)
    df[col]=df[col].clip(lo,hi)
    return df

def handle_missing(df, strategy="median"):
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "most_frequent":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                raise ValueError("strategy must be one of: median,mean,most_frequent")
             
    return df
    
if __name__=="__main__":
   df=load_data()
   nan_cols={"MedInc":0.05,"AveRooms":0.10}
   outlier_cols=["AveOccup"]
   scale_cols=["Population"]
   df_dirty=make_dirty(df,nan_cols,outlier_cols,scale_cols)
   print("Dirty summary")
   print(df_dirty.isna().mean().sort_values(ascending=False).head(5))

   df_fix=handle_missing(df_dirty,strategy="median")
   for c in outlier_cols:
      df_fix=clip_outliers(df_fix,c)


    
    

