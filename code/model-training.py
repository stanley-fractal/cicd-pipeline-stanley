
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestClassifier

#Test for cicd - 1
df = pd.read_csv('data/ecommerce-dataset.csv')
df.dropna(inplace=True)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Reached.on.Time_Y.N', axis=1)
y = df['Reached.on.Time_Y.N']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Save the trained model
model_filename = 'ecommerce-model.pkl'
joblib.dump(model, 'app/ecommerce-model.pkl')
print(f"Model saved as {model_filename}")


