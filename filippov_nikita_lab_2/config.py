# Генерация исходных данных
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

NP_RANDOM_STATE = 0
COUNT_ROW = 750
COUNT_FEATURES = 14
COUNT_DEPENDENT = 4

# Конфигурация моделей
LASSO_ALPHA = 0.1
RFE_COUNT_FEATURES = 5

# RFE_ESTIMATOR = Ridge(alpha=0.1)
RFE_ESTIMATOR = LinearRegression()
# RFE_ESTIMATOR = Lasso(alpha=0.1)

