import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
df = pd.read_csv("cardetails.csv")
print(df.head())
#missing values
print("\nChecking for missing values")
print(df.isnull().sum())
df = df.dropna()
categorical_columns=["Brand","Fuel","Transmission","Owner"]
label_encoders = {}
for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder
print("\nCategorical Encoding completed")
#features and target
X=df.drop("Price", axis=1)
y=df["Price"]
#training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Training samples:",len(X_train))
print("\n Testing samples:",len(X_test))
#train random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("\nTraining the model")
model.fit(X_train, y_train)
print("\nModel training completed")
#prediction
y_pred = model.predict(X_test)
#evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse=mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("\nModel Performance ")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
#save the model
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\nModel saved as car_price_model.pkl")
#save the label encoders
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)
print("\nLabel encoders saved as label_encoders.pkl")
#feature importance
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importances)
print("\nTraining completed successfully.")