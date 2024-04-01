import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm

# Load advertising data (replace 'advertising_data.csv' with your actual file name)
advertising_data = pd.read_csv('Advertising.csv')

# Separate predictor variables (X) and target variable (y)
X = advertising_data[['TV', 'radio', 'newspaper']]  # assuming 'TV', 'radio', 'newspaper' are advertising channels
y = advertising_data['sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
# Fit the model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Predict on test set
linear_reg_predictions = linear_reg_model.predict(X_test)

# Evaluate model
linear_reg_r2 = r2_score(y_test, linear_reg_predictions)

# Calculate residuals
residuals = y_test - linear_reg_predictions

# Calculate residual standard deviation
residual_std = np.std(residuals)

# Calculate t-values
t_values = linear_reg_model.coef_ / (residual_std / np.sqrt(X_train.shape[0] - X_train.shape[1] - 1))

# Calculate p-values
p_values = [2 * (1 - stats.t.cdf(np.abs(t), X_train.shape[0] - X_train.shape[1] - 1)) for t in t_values]

# Print results
print("Linear Regression:")
print("R-squared (R^2):", linear_reg_r2)
print("T-values:", t_values)
print("P-values:", p_values)


# Multiple Regression
# Add intercept term to predictor variables
X_train_multi = sm.add_constant(X_train)
X_test_multi = sm.add_constant(X_test)

# Fit the model
multiple_reg_model = sm.OLS(y_train, X_train_multi).fit()

# Predict on test set
multiple_reg_predictions = multiple_reg_model.predict(X_test_multi)

# Evaluate model
multiple_reg_r2 = r2_score(y_test, multiple_reg_predictions)

# Print Multiple Regression results
print("\nMultiple Regression:")
print("R-squared (R^2):", multiple_reg_r2)
print("T-values:")
print(multiple_reg_model.tvalues)
print("P-values:")
print(multiple_reg_model.pvalues)

