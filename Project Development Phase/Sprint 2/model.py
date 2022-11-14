import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboostreg
import pickle


#Data Loading
df = pd.read_csv("./CarResaleValue.csv")
#printing first 5 rows of dataset
#print(df.head())
#Printing no. of Rows and Columns in the dataset
print(df.shape)

# print(df.describe())# only shows numerical column
df = df.drop(columns=['id'],axis=1)

# print(df.info())

#sns.boxplot(df.yr_mfr)
q1 = df.yr_mfr.quantile(0.25)
q3 = df.yr_mfr.quantile(0.75)
IQR = q3-q1
upper_limit = q3 + 1.5*IQR
lower_limit = q1 - 1.5*IQR
df['yr_mfr']=np.where(df['yr_mfr']<lower_limit,df['yr_mfr'].median(),df['yr_mfr'])
#sns.boxplot(df.yr_mfr)

le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == object or df[column].dtype == bool:
        df[column] = le.fit_transform(df[column].astype(str))
#print(df.head())
#Check whether any null values present in the dataset
# print(df.isnull().any())
df["orginal_price"].fillna(df["sale_price"]+(df["sale_price"] * 0.2), inplace=True)

# print(df.isnull().any())
# df=df.drop(columns=['sale_price'],axis=1)
df= df.drop(columns=['reserved'],axis=1)
# print(df.corr().sale_price.sort_values(ascending=False))

y=df['sale_price']
X = df.drop(columns=['sale_price'],axis=1)
# print(X.shape)
print(list(X.columns.values))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# train the before FS XGBoost model
params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'reg_lambda': 1.5,
    'gamma': 0.0,
    'min_child_weight': 25,
    'base_score': 0.0,
    'tree_method': 'exact',
}
num_boost_round = 50
model_scratch = xgboostreg.XGBoostModel(params, random_seed=0)

class SquaredErrorObjective():
    def loss(self, y, pred): return np.mean((y - pred)**2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))
model_scratch.fit(X_train, y_train, SquaredErrorObjective(), num_boost_round)

pred_scratch = model_scratch.predict(X_test)
#print('R2 Score:',r2_score(pred_scratch,y_test))


xgb_r2=r2_score(pred_scratch,y_test)

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
y_pred= rf_reg.predict(X_test)

rfr_r2=r2_score(y_pred,y_test)

if rfr_r2>xgb_r2:
    pickle.dump(rf_reg, open('model.pkl','wb'))
else:
    pickle.dump(model_scratch, open('model.pkl','wb'))
    
"""mae=[]
mse=[]
rmse=[]
dict_r2={"Before FS": bfs}
print("R2 score:")
print ("{:<20} {:<20} {:<15}".format("    ","XGBoost Regressor","Random Forest Reg"))
for k, v in dict_r2.items():
    v1, v2 = v
    print ("{:<20} {:<20} {:<15}".format(k, v1, v2))
print("--------------------------------------------------------")
mae.append(mean_absolute_error(y_test, y_pred))
mse.append(mean_squared_error(y_test, y_pred))
rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
mae.append(mean_absolute_error(y_test, pred_scratch))
mse.append(mean_squared_error(y_test, pred_scratch))
rmse.append(np.sqrt(mean_squared_error(y_test, pred_scratch)))
print("******************************************************")
d = {"Mean Absolute Error": mae,"Mean Square Error": mse,"RootMeanSquareError": rmse}
print("ERROR Table:")
print ("{:<20} {:<20} {:<15}".format("    ","Random Forest Reg","XGBoost Regressor"))
for k, v in d.items():
    ra, xg = v
    print ("{:<20} {:<20} {:<15}".format(k, ra, xg))

print("******************************************************")"""