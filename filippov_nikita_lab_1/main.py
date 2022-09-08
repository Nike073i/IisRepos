from config import *
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, Perceptron, Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def create_model(model_type, x, y):
    if model_type == 'ridge':
        poly_features = PolynomialFeatures(degree=RIDGE_DEGREE)
        ridge = Ridge(alpha=RIDGE_ALPHA)
        model = Pipeline([('poly', poly_features), (model_type, ridge)])
    elif model_type == 'perceptron':
        model = Perceptron()
    else:
        model = LinearRegression()
    model.fit(x, y)
    return model


data_points, data_classes = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES,
                                                n_redundant=N_REDUNDANT, n_informative=N_INFORMATIVE,
                                                random_state=CF_RANDOM_STATE, n_clusters_per_class=N_CLUSTERS_PER_CLASS)

x_points = data_points[:, 0].reshape(-1, 1)
y_points = data_points[:, 1].reshape(-1, 1)

point_train, point_test, class_train, class_test = train_test_split(data_points, data_classes, test_size=TEST_SIZE)
x_train, x_test, y_train, y_test = train_test_split(x_points, y_points, test_size=TEST_SIZE)

linear_point_model = create_model('linear', x_train, y_train)
ridge_point_model = create_model('ridge', x_train, y_train)

linear_class_model = create_model('linear', point_train, class_train)
ridge_class_model = create_model('ridge', point_train, class_train)
perceptron_model = create_model('perceptron', point_train, class_train)

sorted_x_points = sorted(x_points)
colors = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(x_points, y_points, c=data_classes, cmap=colors)

plt.plot(sorted_x_points, linear_point_model.predict(sorted_x_points), color='green', label='linear')
print(linear_point_model.score(x_test, y_test))

plt.plot(sorted_x_points, ridge_point_model.predict(sorted_x_points), color='orange', label='ridge')
print(ridge_point_model.score(x_test, y_test))

plt.legend()
plt.show()