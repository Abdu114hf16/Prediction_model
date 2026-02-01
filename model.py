import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

raw_housing = fetch_california_housing()
housing_data = pd.DataFrame(raw_housing.data, columns=raw_housing.feature_names)
housing_data['TargetPrice'] = raw_housing.target

inputs = housing_data.drop('TargetPrice', axis=1)
targets = housing_data['TargetPrice']

train_x, val_x, train_y, val_y = train_test_split(inputs, targets, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_x, train_y)

price_predictions = rf_model.predict(val_x)

rmse_val = np.sqrt(mean_squared_error(val_y, price_predictions))
accuracy_score = r2_score(val_y, price_predictions)

print(f"Model RMSE: {rmse_val:.4f}")
print(f"Model Accuracy (R2 Score): {accuracy_score:.2%}")
