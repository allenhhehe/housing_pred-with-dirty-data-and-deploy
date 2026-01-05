import pandas as pd
from sklearn.datasets import fetch_california_housing

ds1=fetch_california_housing(as_frame=True)
df1=ds1.frame

print(df1.head())
print(df1.info())
print(df1.describe())


def load_data(as_frame:bool=True)->pd.DataFrame:
            
    ds=fetch_california_housing(as_frame=as_frame)
    df=ds.frame.copy()
    #sklearn list name offine use "MedhouseVal"
    df.rename(columns={"MedHouseVal":"target"},inplace=True)
    return df