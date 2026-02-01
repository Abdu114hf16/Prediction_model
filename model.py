import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RealEstateModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def preprocess(self):
        self.data.fillna(0, inplace=True)
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        return train_test_split(X, y, test_size=0.2)

    def train(self):
        X_train, X_test, y_train, y_test = self.preprocess()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(f"Model RMSE: {mean_squared_error(y_test, predictions, squared=False)}")

if __name__ == "__main__":
    print("Initializing Real Estate Predictive Model...")
