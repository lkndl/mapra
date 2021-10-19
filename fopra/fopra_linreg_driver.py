#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import scipy.stats
import seaborn as sns
from numpy.random import default_rng
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(str(Path('.').resolve().parent))

from mapra import prep

WD = Path(__file__).resolve().parents[1]  # this is supposed to mean 'working directory'
sns.set_theme(style='white')
rng = np.random.default_rng(12345)


def train_predictors(splits, random_state=None):
    """
    Train
    make line plots showing the dependency on the dataset size

    :param splits:
    :param random_state:
    :return:
    """
    if random_state is None:
        random_state = rng.integers(low=0, high=1000, size=1)[0]
    local_rng = np.random.default_rng(random_state)

    # train from subsets of different sizes
    relative_dataset_sizes = np.arange(.05, 1.01, .5)  # .05)
    # shuffle so runtime will statistically not be changed by other jobs
    local_rng.shuffle(relative_dataset_sizes)

    results = np.zeros((len(relative_dataset_sizes) * 3 * 4, 14))

    for row, set_size in enumerate(relative_dataset_sizes):
        for i, (delta, split) in enumerate(splits.items()):
            print(f'{random_state} {row} {i} {delta} {set_size} ', end='')

            # to select a subset of set_size, re-combine features and labels
            shared = np.hstack((split.y, split.X)).copy()  # make a copy to be sure splits isn't changed
            # shuffle the dataset
            local_rng.shuffle(shared)
            y, X = shared[:, 0].reshape(-1, 1), shared[:, 1:]

            # calculate number of leading rows to pick
            abs_set_size = int(max(1, np.round(set_size * X.shape[0])))
            print(f'dataset size: {abs_set_size} ', end='')
            X, y = X[:abs_set_size, :], y[:abs_set_size, :]

            for mode in range(4):
                # the different methods have to be the innermost loop to use identical data
                # if anything other than LassoLarsCV use LassoLarsIC(criterion='aic') or LassoCV(cv= something high
                if mode == 0:
                    print(f'LassoLarsCV ', end='')
                    regr = linear_model.LassoLarsCV(n_jobs=-1, copy_X=True)
                    regr.fit(X, y.flatten())
                    n_cols = len(regr.active_)
                elif mode == 1:
                    print(f'LassoLarsIC ', end='')
                    regr = linear_model.LassoLarsIC(criterion='aic', copy_X=True)
                    regr.fit(X.astype(float), y.flatten().astype(float))
                    n_cols = sum([1 for i in regr.coef_ if i != 0])
                elif mode == 2:
                    print(f'LarsCV ', end='')
                    regr = linear_model.LarsCV(max_iter=100, n_jobs=-1, copy_X=True)
                    regr.fit(X, y.flatten())
                    n_cols = len(regr.active_)
                # elif mode == 3:
                #     print('LassoLars', end='')
                #     regr = linear_model.LassoLars(alpha=.1)
                #     regr.fit(X.astype(float), y.flatten().astype(float))
                #     n_cols = len(regr.active_)
                elif mode == 3:
                    print(f'LinearRegression ', end='')
                    regr = linear_model.LinearRegression(n_jobs=-1, copy_X=True)
                    regr.fit(X.astype(float), y.flatten().astype(float))
                    n_cols = sum([1 for i in regr.coef_ if i != 0])
                # elif mode == 5:
                #     print(f'LassoCV ', end='')
                #     regr = linear_model.LassoCV(n_jobs=-1, max_iter=1000)
                #     regr.fit(X.astype(float), y.flatten().astype(float))
                #     print(dir(regr))
                #     n_cols = 0
                # LassoCV(alphas=np.arange(.001, .1, .01))  # Lasso(alpha=0.01)

                alpha = regr.alpha_ if mode != 3 else 0
                print(f'non-zero: {n_cols} alpha: {alpha} ', end='')

                # make a prediction
                y_pred = regr.predict(split.X_test).reshape(-1, 1)
                ar = np.hstack((y_pred, split.y_true))
                # evaluate the prediction
                sp, pval = scipy.stats.spearmanr(ar, axis=0)
                pcorr = np.corrcoef(ar, rowvar=False)[0, 1]
                rmse = mean_squared_error(split.y_true, y_pred, squared=False)
                r2 = r2_score(split.y_true, y_pred)

                line_idx = row * 3 + 3 * mode * len(relative_dataset_sizes) + i
                print(f'line: {line_idx}')
                results[line_idx, :] = \
                    i, abs_set_size, n_cols, rmse, sp, pval, alpha, r2, pcorr, \
                    seed, mode, set_size, split.real_test_size, split.records_test_size

    return results


seeds = [0, 1, 2, 3, 4, 5, 6, 1024, 1025, 1026, 1027, 1028, 1029]
seeds = [0, 1, 2]

if __name__ == '__main__':
    data = prep.protherm_dataset(WD)
    df = data.dataframe_abbrev({'DATASET': 'reduced_set'})
    (WD / 'txts').mkdir(parents=True, exist_ok=True)

    # train repeatedly with different validation sets
    for j, seed in enumerate(seeds):
        splits = data.uniprot_train_test_split(df=df, test_size=.2, random_state=seed)
        result = train_predictors(splits, random_state=seed)
        np.save(str(WD / 'txts' / f'fopra_{seed}.npy'), result)

    print('concatenating array ...', end='')
    ar = np.vstack([np.load(str(WD / 'txts' / f'fopra_{seed}.npy'),
                            allow_pickle=False) for seed in seeds])
    np.save(str(WD / 'txts' / f'fopra_all.npy'), ar)
    print(f'saved array for all seeds: {seeds}')
    [(WD / 'txts' / f'fopra_{seed}.npy').unlink() for seed in seeds]
