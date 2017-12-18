from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X_train = [[1,2],[2,4],[6,7]]
y_train = [1.2, 4.5, 6.7]
X_test = [[1,3],[2,5]]

# create a Linear Regressor
lin_regressor = LinearRegression()

# pass the order of your polynomial here
poly = PolynomialFeatures(2)

# convert to be used further to linear regression
X_transform = poly.fit_transform(X_train)

print(X_transform)

# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train)

# get the predictions
lin_regressor.fit(X_transform, y_train)