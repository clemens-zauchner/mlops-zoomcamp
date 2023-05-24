import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df_jan = pd.read_parquet("week1/data/yellow_tripdata_2022-01.parquet")

print("Number of columns:", len(df_jan.columns))

def preprocess_data(df):
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)
    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).astype("timedelta64[m]")
    df = df.loc[df["duration"].between(1, 60)]
    return df


df_jan_preprocessed = preprocess_data(df_jan)
print("Standard deviation of the trip duration:", df_jan_preprocessed["duration"].std().round(2))
print("Percentage of records left after dropping outliers:", df_jan_preprocessed.shape[0]/df_jan.shape[0])

df_dict = df_jan_preprocessed[["PULocationID", "DOLocationID"]].astype(str).to_dict(orient="records")
dv = DictVectorizer()
X_train = dv.fit_transform(df_dict)
y_train = df_jan_preprocessed["duration"]

print("Number of columns:", X_train.shape[1])

lr = LinearRegression().fit(X_train, y_train)

y_pred = lr.predict(X_train)

rmse = mean_squared_error(y_train, y_pred, squared=False)

print("RMSE training:", rmse)

df_feb = pd.read_parquet("week1/data/yellow_tripdata_2022-02.parquet")

df_feb_preprocessed = preprocess_data(df_feb)

df_dict = df_feb_preprocessed[["PULocationID", "DOLocationID"]].astype(str).to_dict(orient="records")
X_val = dv.transform(df_dict)
y_val = df_feb_preprocessed["duration"]

y_pred_val=lr.predict(X_val)

rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

print("RMSE validation:", rmse_val)
