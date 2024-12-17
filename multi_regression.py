import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


df0 = pd.read_csv("dataset.csv")

# Convert `tpep_pickup_datetime` and `tpep_dropoff_datetime` to datetime format
df0['tpep_pickup_datetime'] = pd.to_datetime(df0['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df0['tpep_dropoff_datetime'] = pd.to_datetime(df0['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Calculate the duration in minutes
df0['duration'] = (df0['tpep_dropoff_datetime'] - df0['tpep_pickup_datetime']) / np.timedelta64(1, 'm')

# Visualize boxplots for outlier detection
fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for Outlier Detection')
sns.boxplot(ax=axes[0], x=df0['trip_distance'])
sns.boxplot(ax=axes[1], x=df0['fare_amount'])
sns.boxplot(ax=axes[2], x=df0['duration'])
plt.show()

# Handling fare_amount Outliers
# Calculate Q1, Q3, and IQR
Q1 = df0['fare_amount'].quantile(0.25)
Q3 = df0['fare_amount'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound_fare = Q1 - 1.5 * IQR
upper_bound_fare = Q3 + 1.5 * IQR
print(f"Fare Amount Outlier Bounds - Lower: {lower_bound_fare}, Upper: {upper_bound_fare}")

# Cap fare_amount values outside the bounds
df0['fare_amount'] = np.where(df0['fare_amount'] < 0, 0, df0['fare_amount'])
df0['fare_amount'] = np.where(df0['fare_amount'] > upper_bound_fare, upper_bound_fare, df0['fare_amount'])

# Handling duration Outliers
# Calculate Q1, Q3, and IQR for duration
Q1_duration = df0['duration'].quantile(0.25)
Q3_duration = df0['duration'].quantile(0.75)
IQR_duration = Q3_duration - Q1_duration

# Define bounds for duration
lower_bound_duration = Q1_duration - 1.5 * IQR_duration
upper_bound_duration = Q3_duration + 1.5 * IQR_duration

# Cap duration outliers
df0['duration'] = np.where(df0['duration'] < lower_bound_duration, lower_bound_duration, df0['duration'])
df0['duration'] = np.where(df0['duration'] > upper_bound_duration, upper_bound_duration, df0['duration'])

# Build the Multiple Linear Regression Model
# Select Features and Target
X = df0[['trip_distance', 'duration', 'passenger_count', 'tolls_amount']]  # Example features
y = df0['fare_amount']

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Evaluate the Model Performance
print("Training Data Evaluation")
r_sq_train = lr.score(X_train_scaled, y_train)
y_pred_train = lr.predict(X_train_scaled)
print('R^2 (Training):', r2_score(y_train, y_pred_train))
print('MAE (Training):', mean_absolute_error(y_train, y_pred_train))
print('MSE (Training):', mean_squared_error(y_train, y_pred_train))
print('RMSE (Training):', np.sqrt(mean_squared_error(y_train, y_pred_train)))

print("\nTesting Data Evaluation")
y_pred_test = lr.predict(X_test_scaled)
print('R^2 (Testing):', r2_score(y_test, y_pred_test))
print('MAE (Testing):', mean_absolute_error(y_test, y_pred_test))
print('MSE (Testing):', mean_squared_error(y_test, y_pred_test))
print('RMSE (Testing):', np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Visualize Results
# Create a results dataframe
results = pd.DataFrame({'actual': y_test, 'predicted': y_pred_test})
results['residual'] = results['actual'] - results['predicted']

# Scatterplot: Predicted vs Actual
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x='actual', y='predicted', data=results, alpha=0.5, ax=ax)
plt.plot([0, max(results['actual'])], [0, max(results['actual'])], color='red', linestyle='--')
plt.title('Actual vs Predicted')
plt.show()

# Histogram of Residuals
sns.histplot(results['residual'], bins=30, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Scatterplot: Residuals vs Predicted
sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted')
plt.show()

# Model Coefficients
coefficients = pd.DataFrame(lr.coef_, index=X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

# Interpret the `trip_distance` coefficient
mean_distance_sd = X_train['trip_distance'].std()
mean_distance_coeff_scaled = coefficients.loc['trip_distance', 'Coefficient']
mean_distance_coeff_unscaled = mean_distance_coeff_scaled / mean_distance_sd
print(f"\nUnscaled Coefficient for Trip Distance: {mean_distance_coeff_unscaled:.2f} (per mile)")
