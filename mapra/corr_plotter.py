#!/usr/bin/env python3

from itertools import product
from pathlib import Path

import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import paired_distances

order = ['dtemp', 'ddg', 'h2o']
simple_lookup = {'∆Tm_(C)': 'dtemp', '∆∆G_(kcal/mol)': 'ddg', '∆∆G_H2O_(kcal/mol)': 'h2o'}
tex_lookup = {'dtemp': 'ΔT$_{\mathrm{m}}$', 'ddg': 'ΔΔG', 'h2o': 'ΔΔG$_{\mathrm{H_2O}}$',
              '∆Tm_(C)': 'ΔT$_{\mathrm{m}}$', '∆∆G_(kcal/mol)': 'ΔΔG',
              '∆∆G_H2O_(kcal/mol)': 'ΔΔG$_{\mathrm{H_2O}}$', 'delta_g': 'ΔΔG',
              'melttemp': 'ΔT$_{\mathrm{m}}$', 'delta_g_h2o': 'ΔΔG$_{\mathrm{H_2O}}$', }

wd = Path('.').resolve().parent


def make_corr_plot(mbeds, df, extend, func, modify, scaler):
    pdists = dict()
    pairwise_metrics = ['euclidean', 'cosine', 'manhattan']

    for uniprot_id, d in mbeds.items():
        wt = d.pop('wt')
        try:
            # make sure that even if something goes wrong, we put back the wildtype
            pdists[uniprot_id] = dict()
            for variant, ar in d.items():
                positions = [int(p[:-1]) - 1 for p in variant.split('_')]

                # extend to neighbourhood
                positions = sorted(set([c for ran in [list(range(
                    max(0, p - extend), min(len(wt), p + extend + 1)))
                    for p in positions] for c in ran]))

                pdists[uniprot_id][variant] = {m: func(
                    paired_distances(wt[positions, :], ar, metric=m)) for m in pairwise_metrics}
        except Exception as ex:
            print(ex)
        d['wt'] = wt

    for m in pairwise_metrics:
        if modify == 'flip':
            # if the measured change is negative, pretend the distance is negative
            df[m] = df.apply(lambda gdf: np.sign(gdf[gdf.DELTA]) * pdists.get(
                gdf.UniProt_ID, dict()).get(gdf.MUTATION, dict()).get(m, 0), axis=1)
        else:
            df[m] = df.apply(lambda gdf: pdists.get(
                gdf.UniProt_ID, dict()).get(gdf.MUTATION, dict()).get(m, 0), axis=1)
        if scaler == 'std':
            me, std = np.mean(df[m]), np.std(df[m])
            df[m] = df[m].apply(lambda f: (f - me) / std)
        elif scaler == 'minmax':
            lower, upper = 0, 1
            mi, ma = min(df[m]), max(df[m])
            df[m] = df[m].apply(lambda f: (f - mi) / (ma - mi) * (upper - lower) + lower)

    # tranform the dataframe
    dfc = df.loc[:, order + pairwise_metrics].melt(
        id_vars=order, value_vars=pairwise_metrics,
        var_name='metric', value_name='dist').melt(
        id_vars=['metric', 'dist'], value_vars=order,
        var_name='delta', value_name='change')
    dfc = dfc[~dfc.change.isna()].reset_index(drop=True)  # drop all the NaN lines. it's ok why they were there

    if modify == 'abs':
        dfc.change = dfc.change.abs()  # make all changes positive
    elif modify == 'pos':
        dfc = dfc.loc[dfc.change > 0]
    elif modify == 'neg':
        dfc = dfc.loc[dfc.change < 0]

    # calculate Pearson correlation coefficient
    covs = dict()
    for d, m in product(order, pairwise_metrics):
        covs[d, m] = np.corrcoef(
            dfc.loc[(dfc.delta == d) & (dfc.metric == m),
                    ['dist', 'change']], rowvar=False)[0, 1]
    print(covs)

    # create a plot
    g = sns.lmplot(data=dfc,
                   x='dist', y='change',
                   hue='delta', palette='viridis',
                   sharey='row',
                   col='metric', col_order=pairwise_metrics,
                   row='delta', row_order=order,
                   scatter_kws={'s': 25, 'alpha': .1},
                   height=3.6, aspect=1, ci=95, order=1,  fit_reg=False,
                   )

    for (d, m), ax in g.axes_dict.items():
        # ax.axline((0 ,0), slope=1, color='.5', lw=.7)
        pearson_corr = np.corrcoef(dfc.loc[(dfc.delta == d) & (dfc.metric == m),
                                           ['dist', 'change']], rowvar=False)[0, 1]
        ax.text(.07, .9, f'{pearson_corr:.2f}', transform=ax.transAxes)
    for i, ax in enumerate(g.axes[2,]):
        ax.set_xlabel(pairwise_metrics[i])
    for i, ax in enumerate(g.axes[:, 0]):
        ax.set_ylabel(tex_lookup[order[i]] + [' [°C]', ' [kcal/mol]', ' [kcal/mol]'][i])

    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title('')
        if scaler == 'minmax' and modify != 'flip':
            ax.set_xlim(0, 1)
            ax.set_xticks([.2, .4, .6, .8, 1])
            ax.set_xticklabels(['.2', '.4', '.6', '.8', '1'])

    g.fig.subplots_adjust(hspace=.1)

    (wd / 'plots' / 'corrs').mkdir(parents=True, exist_ok=True)
    g.savefig(wd / 'plots' / 'corrs' / f'new_{extend}_{modify}_{func.__name__}_{scaler}.png', dpi=300)
