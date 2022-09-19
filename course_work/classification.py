from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from config import *

classifier_tree = DecisionTreeClassifier(random_state=DTC_RANDOM_STATE)
x = []
y = []


def estimate(test_size, precision):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=TRAIN_RANDOM_STATE)
    classifier_tree.fit(x_train, y_train)
    test_score = classifier_tree.score(x_test, y_test)
    feature_scores = map(lambda score: round(score, precision), classifier_tree.feature_importances_)
    return test_score, dict(zip(FEATURE_COLUMNS, feature_scores))
