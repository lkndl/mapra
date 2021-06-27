#!/usr/bin/env python3

import pickle
from pathlib import Path

import numpy as np
import scipy.stats
import seaborn as sns
from numpy.random import default_rng
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import prep

WD = Path(__file__).resolve().parents[1]  # this is supposed to mean 'working directory'
(WD / 'txts').mkdir(parents=True, exist_ok=True)

sns.set_theme(style='white')
rng = np.random.default_rng(12345)

data = prep.dataset(WD)
npr = data.fetch_numpy_distances()
spearman_rhos = data.fetch_spearman_rhos(npr)
delta_labels = data.order
rng = np.random.default_rng(12345)


def make_splits(npr, seed=rng.integers(low=0, high=1000, size=1)[0]):
    # print(f'split seed {seed}')
    # create validation sets
    splits = dict()
    test_size = .2
    for i, delta in enumerate(delta_labels):
        # select the rows for this delta, and cleave off the delta column
        dnpr = npr[npr[:, 0] == i, 1:]
        # split into features and labels
        X, y = dnpr[:, 1:], dnpr[:, 0].reshape(-1, 1)
        # split into test and training data
        splits[delta] = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        # X, X_test, y, y_true
    return splits


def dodo(splits, seed=rng.integers(low=0, high=1000, size=1)[0]):
    relative_dataset_sizes = [1]
    rng.shuffle(relative_dataset_sizes)
    # print(f'train seed {seed}')

    all_coefs = list()
    all_spears = list()

    results = np.zeros((len(relative_dataset_sizes) * 3, 9))
    for row, set_size in enumerate(relative_dataset_sizes):
        for i, delta in enumerate(delta_labels):
            print(f'{seed} {row} {i} {delta} ', end='')

            X, X_test, y, y_true = splits[delta]

            # shuffle the dataset
            shared = np.hstack((y, X))
            rng.shuffle(shared)
            y, X = shared[:, 0].reshape(-1, 1), shared[:, 1:]

            # calculate number of leading rows to pick
            abs_set_size = int(max(1, np.round(set_size * X.shape[0])))
            print(f'dataset size: {abs_set_size} ', end='')
            X, y = X[:abs_set_size, :], y[:abs_set_size, :]

            regr = linear_model.LassoLarsCV(n_jobs=-1)
            # LassoCV(alphas=np.arange(.001, .1, .01))  # Lasso(alpha=0.01)
            regr.fit(X, y.flatten())
            n_cols = len(regr.active_)
            print(f'non-zero: {n_cols} alpha: {regr.alpha_}')

            # make a prediction
            y_pred = regr.predict(X_test).reshape(-1, 1)
            ar = np.hstack((y_pred, y_true))
            # evaluate the prediction
            sp, pval = scipy.stats.spearmanr(ar, axis=0)
            pcorr = np.corrcoef(ar, rowvar=False)[0, 1]
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            r2 = r2_score(y_true, y_pred)

            # find the best-correlating columns in the training set
            spears = scipy.stats.spearmanr(np.hstack((y, X)), axis=0)[0][0, 1:]
            spears = sorted(np.argpartition(abs(spears), -n_cols)[:-n_cols - 1:-1])

            results[row * 3 + i, :] = i, abs_set_size, n_cols, rmse, sp, pval, regr.alpha_, r2, pcorr
            all_spears.append(spears)
            all_coefs.append(regr.coef_)

    # np.save(str(WD / 'txts' / f'{seed}_fresh.npy'), results)
    return results, all_coefs, all_spears


results = list()
all_coefs = dict()
all_spears = dict()
for seed in range(1000):
    splits = make_splits(npr.copy(), seed)
    result, coefs, spears = dodo(splits, seed)
    results.append(result)
    ar = np.vstack(results)
    all_coefs[seed] = coefs
    all_spears[seed] = spears

    if not seed % 10 or seed == 999:
        np.save(str(WD / 'txts' / f'{seed // 100}.npy'), ar)
        with open(WD / 'txts' / f'all_coefs_{seed // 100}.pkl', 'wb') as file:
            pickle.dump(all_coefs, file)
        with open(WD / 'txts' / f'all_spears_{seed // 100}.pkl', 'wb') as file:
            pickle.dump(all_spears, file)

results = [dodo(make_splits(npr.copy(), seed), seed) for seed in range(1000)]
