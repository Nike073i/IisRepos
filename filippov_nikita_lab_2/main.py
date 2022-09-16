from sklearn.feature_selection import RFE, r_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import f_regression
from operator import itemgetter
from config import *
import numpy as np


def ranks_to_dict(var_rank, var_names):
    minmax = MinMaxScaler()
    normalized_ranks = minmax.fit_transform(np.array(var_rank).reshape(-1, 1)).ravel()
    normalized_ranks = map(lambda x: round(x, 2), normalized_ranks)
    return dict(zip(var_names, normalized_ranks))


def generate_data(count_row, count_options, count_dependent):
    np.random.seed(NP_RANDOM_STATE)
    x = np.random.uniform(0, 1, (count_row, count_options))

    # F(x) = 10*sin(Pi*x0*x1)+20*(x3-0.5)^2 + 10*x4+5*x5^5 + rand
    y = (10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 +
         10 * x[:, 3] + 5 * x[:, 4] ** 5 + np.random.normal(0, 1))

    x[:, 10:] = x[:, :4] + np.random.normal(0, .025, (count_row, count_dependent))

    return x, y


def get_mean_ranks(dict_f_ranks):
    m_mean_ranks = dict()
    for model_name, feature_ranks in dict_f_ranks.items():
        for name, rank in feature_ranks.items():
            if name not in m_mean_ranks:
                m_mean_ranks[name] = 0
            m_mean_ranks[name] += rank
    for feature_name, feature_rank in m_mean_ranks.items():
        mean_rank = feature_rank / len(dict_f_ranks)
        m_mean_ranks[feature_name] = round(mean_rank, 2)
    return m_mean_ranks


X, Y = generate_data(COUNT_ROW, COUNT_FEATURES, COUNT_DEPENDENT)
feature_names = ["x%s" % (i + 1) for i in range(0, COUNT_FEATURES)]

# Лассо
lasso = Lasso(alpha=LASSO_ALPHA)
lasso.fit(X, Y)

# # Рекурсивное сокращение признаков
rfe = RFE(RFE_ESTIMATOR, n_features_to_select=RFE_COUNT_FEATURES)
rfe.fit(X, Y)

# Линейная корреляция
F, pv = f_regression(X, Y)

model_ranks = {
    'Lasso': ranks_to_dict(lasso.coef_, feature_names),
    'RFE': ranks_to_dict(rfe.support_, feature_names),
    'Linear': ranks_to_dict(F, feature_names)
}

mean_ranks = get_mean_ranks(model_ranks)
sorted_mean_ranks = sorted(mean_ranks.items(), key=itemgetter(1), reverse=True)

print("Lasso feature ranks = ", model_ranks["Lasso"], '\n')
print("RFE feature ranks = ", model_ranks["RFE"], '\n')
print("Linear correlation feature ranks = ", model_ranks["Linear"], '\n')
print("Mean feature ranks = ", sorted_mean_ranks, '\n')
