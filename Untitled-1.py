# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df=pd.read_csv("retail_store_inventory.csv")

df.info()
df.describe()
df.head()


# %%
#handling missing ,duplicate value and check correct values
df.isnull().sum()
df.duplicated().sum()
df["Demand Forecast"]=df["Demand Forecast"].apply(lambda x:max(x,0))
df.info()

# %%
#feature engneering
df["Date"]=pd.to_datetime(df["Date"])

df["Year"]=df["Date"].dt.year
df["Quarter"]=df["Date"].dt.quarter
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df["Day"]=df['Date'].dt.day

df['Weekday'] = df['Date'].dt.day_name()
df["Is_Weekend"]=df["Date"].dt.weekday.apply(lambda x:1 if x>=5 else 0)
df.info()




# %%
def Season(month):
  if month in [12,1,2]:
    return 'Spring'
  elif month in [3,4,5]:
    return 'Winter'
  elif month in [6,7,8]:
    return 'Autumn'
  else:
    return 'Summer'
  
df["Season"]=df['Month'].apply(Season)
df["Season"].sample(5)

# %%
#lagging feature to capute past behaviour and to predict future
df_original=df.copy()
df = df_original.copy()  # use your original dataset
df=df.sort_values(["Store ID","Product ID","Date"])


df["Sales_lag_1"]=df.groupby(["Store ID","Product ID"])['Units Sold'].shift(1)
df["Sales_lag_7"]=df.groupby(["Store ID","Product ID"])["Units Sold"].shift(7)
df["Sales_lag_30"]=df.groupby(["Store ID","Product ID"])["Units Sold"].shift(30)

df['Rolling_mean_7'] = df['Units Sold'].rolling(window=7).mean()
df['Rolling_std_7'] = df['Units Sold'].rolling(window=7).std()
df.fillna(0, inplace=True)
df.info()





# %%
#input features and target
target="Units Sold"
input_features=["Sales_lag_1","Sales_lag_7","Sales_lag_30",
                "Rolling_mean_7","Rolling_std_7",
                "Price","Discount",
                "Holiday/Promotion","Competitor Pricing",
                "Month","Is_Weekend"]

x=df[input_features]
y=df[target]
print(x.shape)
y.shape
df["Date"].min(),df["Date"].max()


# %%
#split the dataset to test and traing dataset based on Date
split_date='2023-10-01'
train=df[df["Date"]<=split_date]
test=df[df["Date"]>'2023-10-01']
X_train=train[input_features]
Y_train=train[target]

X_test=test[input_features]
Y_test=test[target]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# %%
#Train the model with Random forest regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
model=RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_split=5,min_samples_leaf=2,max_features='sqrt',random_state=42, n_jobs=-1)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print(Y_pred[0:10])

# %%
#prediction on test data
Y_pred=model.predict(X_test)

mean_abs_err=mean_absolute_error(Y_test,Y_pred)
mean_squ_err=mean_squared_error(Y_test,Y_pred)
rmae=np.sqrt(mean_squ_err)

mask=Y_test!=0
mape=np.mean(np.abs((Y_test[mask]-Y_pred[mask])/Y_test[mask]))*100

print("mean absolute error:",mean_abs_err)
print("mean square error:",mean_squ_err)
print("root mean square error:",rmae)
print("mean absolute percentage error:",mape)


# %%
#visualization on predictions
plt.figure(figsize=(10,6))
plt.plot(Y_test[:100].reset_index(drop=True),label="Actual")
plt.plot(Y_pred[:100],label="Predicted")

plt.legend()
plt.title("Actual vs Predicted")
plt.show()


