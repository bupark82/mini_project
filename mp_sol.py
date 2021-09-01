import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from seaborn import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

st.title("MPG 연비 예측 App")

df = load_dataset("mpg")

df.drop(columns="name",inplace=True)
df.isnull().sum(axis=0)
df.dropna(axis=0, inplace=True)
df = df.join(pd.get_dummies(df["origin"], drop_first=True)).drop(columns=["origin"])

y_label = ["mpg"]



my_features_X = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "japan", "usa"]

df_X = df.drop(columns="mpg")
column_names = df_X.columns

Y= df["mpg"]

my_scaler = MinMaxScaler()
X_scaled = my_scaler.fit_transform(df_X.values)

df_X_scaled = pd.DataFrame(data=X_scaled, columns= column_names)

st.sidebar.header("차량의 SPEC을 입력해 주세요.")
my_cylinders = st.sidebar.slider('cylinders', float(df_X.cylinders.min()), float(df_X.cylinders.max()), float(df_X.cylinders.mean()))
my_displacement = st.sidebar.slider('displacement', float(df_X.displacement.min()), float(df_X.displacement.max()), float(df_X.displacement.mean()))
my_horsepower = st.sidebar.slider('horsepower', float(df_X.horsepower.min()), float(df_X.horsepower.max()), float(df_X.horsepower.mean()))
my_weight = st.sidebar.slider('weight', float(df_X.weight.min()), float(df_X.weight.max()), float(df_X.weight.mean()))
my_acceleration = st.sidebar.slider('acceleration', float(df_X.acceleration.min()), float(df_X.acceleration.max()), float(df_X.acceleration.mean()))
my_model_year = st.sidebar.slider('model_year', float(df_X.model_year.min()), float(df_X.model_year.max()), float(df_X.model_year.mean()))
my_japan = df_X.japan
my_usa = df_X.usa
my_select_X = st.sidebar.selectbox("origin:", ["Europe","Japan", "USA"])
if my_select_X == "Japan":
    df_X["japan"] = 1
    df_X["usa"] = 0
elif my_select_X == "USA":
    df_X["japan"] = 0
    df_X["usa"] = 1
else:
    df_X["japan"] = 0
    df_X["usa"] = 0

# 입력된 X 데이터.
st.header("입력된 X 데이터:")
my_X_raw = np.array([[my_cylinders, my_displacement, my_horsepower, my_weight, my_acceleration, my_model_year, df_X["japan"][0], df_X["usa"][0]]])
my_df_X_raw = pd.DataFrame(data=my_X_raw, columns=my_features_X)
st.write(my_df_X_raw)

# 전처리된 X 데이터.
with open("my_scaler.pkl","rb") as f:
    my_scaler = pickle.load(f)
my_X_scaled = my_scaler.transform(my_X_raw)     # fit_transform이 아닌 transform!!

st.header("전처리된 X 데이터:")
my_df_X_scaled = pd.DataFrame(data=my_X_scaled, columns=my_features_X)
st.write(my_df_X_scaled)

# 예측.
with open("my_regressor.pkl","rb") as f:
    my_regressor = pickle.load(f)

my_Y_pred = my_regressor.predict(my_X_scaled)

st.header("예측 결과:")
st.write("MPG 예측값:  ", my_Y_pred[0])

my_fig, my_ax = plt.subplots(figsize=(5,3))
my_ax.hist(Y, bins=30, alpha=0.4, color="green")
my_ax.plot([my_Y_pred,my_Y_pred], [0,40], color="red", lw=1, ls=":", alpha=0.8)
my_ax.set_ylim([0,40])
my_ax.set_xlabel("mpg")
my_ax.set_title("MPG Distribution and Yours in Red")
st.pyplot(my_fig)
