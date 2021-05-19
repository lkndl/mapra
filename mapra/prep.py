import h5py
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import norm
from dataclasses import dataclass

from sklearn import linear_model, metrics, preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_distances


def abbrev(mutation_pattern):
    return '_'.join(i[1:] for i in mutation_pattern.split(' '))


@dataclass
class mbed_dists:
    data: pd.DataFrame
    delta_labels: list
    metric_labels: list


class dataset:
    path_regex = re.compile(r'.*?_prothermdb_(?P<measurement>.+?)(?:(?:_)?(?P<dataset>rep_seq|a?)?\.fasta|\.tsv)')
    mutation_regex = re.compile(r'( ?(?:\S+:)?[ARNDCQEGHILKMFPSTWYV]\d{1,9}[ARNDCQEGHILKMFPSTWYV] ?)+')

    def __init__(self, wd=Path('.').resolve().parent, legacy=False):
        print(wd)
        sets = [dict(), dict()]  # full_set, reduced_set
        self.__mbeds__ = None
        self.__dataframe_pairwise__ = None

        for fasta in wd.rglob('*.fasta'):
            m = dataset.path_regex.match(fasta.name)
            if not m:
                warnings.warn('unexpected FASTA filename: ' + str(fasta), RuntimeWarning)
                continue
            gd = m.groupdict()
            annotations = sets[bool(gd['dataset'])]
            ms = gd['measurement']
            if ms in annotations:
                warnings.warn('overwriting sequence data for ' + ms, RuntimeWarning)
            annotations[ms] = {record.id: len(record) for record in SeqIO.parse(fasta, 'fasta')}

        tsvs = list(wd.rglob('*.tsv'))
        if len(tsvs) != 1:
            warnings.warn('found not exactly one TSV with annotations:\n' + '\n'.join(tsvs))

        df = pd.read_csv(tsvs[0], sep='\t')
        # filter out rows with undetermined '-' or 'wild-type' mutation
        len_before = len(df)
        df = df.loc[~df.MUTATION.isin(['-', 'wild-type'])]
        # filter out rows with undetermined UniProt_ID
        df = df.loc[df.UniProt_ID != '-']
        if len_before != len(df):
            warnings.warn('found %d rows with immediately invalid annotation'
                          % (len_before - len(df)), RuntimeWarning)
        df[['MUTATION', 'SOURCE']] = df.MUTATION.str.rstrip(')') \
            .str.split(' \(Based on ', expand=True)
        df['MUT_COUNT'] = df.MUTATION.str.strip().str.count(' ') + 1

        # split rows with multiple measurement into separate records
        metrics = {'∆Tm_(C)', '∆∆G_(kcal/mol)', '∆∆G_H2O_(kcal/mol)'}
        df.melt(id_vars=[c for c in df.columns if c not in metrics], value_vars=metrics)

        df = df.melt(id_vars=[c for c in df.columns if c not in metrics],
                     value_vars=metrics, var_name='DELTA')
        df = df.loc[df.value != '-'].reset_index()

        pivoted = df.pivot(index='index', columns='DELTA', values='value')
        df = df.merge(pivoted, on='index').drop(columns=['index', 'value'])

        translate = lambda m: {'∆Tm_(C)': 'melttemp', '∆∆G_(kcal/mol)': 'delta_g',
                               '∆∆G_H2O_(kcal/mol)': 'delta_g_h2o'}.get(m, 'invalid')

        get_dataset = lambda s: 'reduced_set' if s.UniProt_ID in sets[1][translate(s.DELTA)].keys() \
            else 'full_set' if s.UniProt_ID in sets[0][translate(s.DELTA)].keys() else 'invalid'
        df['DATASET'] = df.apply(get_dataset, axis=1)

        get_length = lambda s: sets[1][translate(s.DELTA)][s.UniProt_ID] if s.DATASET == 'reduced_set' \
            else sets[0][translate(s.DELTA)][s.UniProt_ID] if s.DATASET == 'full_set' else -1
        df['LENGTH'] = df.apply(get_length, axis=1)

        # cut off additional values in parentheses
        for m in metrics:
            df[m] = df[m].str.split('(').str[0].astype(float)

        # def get_repeats_and_std(gdf):
        #     gdf['REPEATS'] = len(gdf)
        #     gdf['STD'] = gdf[gdf.DELTA.iat[0]].std()
        #     return gdf
        # df = df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).apply(get_repeats_and_std)
        # MARK use transform('count') to create a new column and count() otherwise
        # TODO this ignores the pH - that's not real repeats
        df['REPEATS'] = df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).transform('count')['LENGTH']

        self.order = ['dtemp', 'ddg', 'h2o']
        self.simple_lookup = {'∆Tm_(C)': 'dtemp', '∆∆G_(kcal/mol)': 'ddg', '∆∆G_H2O_(kcal/mol)': 'h2o'}
        self.tex_lookup = {'dtemp': 'ΔT$_{\mathrm{m}}$', 'ddg': 'ΔΔG', 'h2o': 'ΔΔG$_{\mathrm{H_2O}}$',
                           '∆Tm_(C)': 'ΔT$_{\mathrm{m}}$', '∆∆G_(kcal/mol)': 'ΔΔG',
                           '∆∆G_H2O_(kcal/mol)': 'ΔΔG$_{\mathrm{H_2O}}$', 'delta_g': 'ΔΔG',
                           'melttemp': 'ΔT$_{\mathrm{m}}$', 'delta_g_h2o': 'ΔΔG$_{\mathrm{H_2O}}$', }

        if legacy:
            # for dataset_intro
            self.full_set_lengths, self.reduced_set_lengths = sets
        else:
            df = df.rename(columns=self.simple_lookup)
            df['DELTA'] = df['DELTA'].apply(self.simple_lookup.get)
            # df = df.drop(columns=['MEASURE', 'METHOD', 'SOURCE', 'T_(C)'])
            df = df.drop(columns=['SOURCE', 'T_(C)'])

        self.__dataframe__ = df.sort_values(by=['UniProt_ID', 'MUTATION']) \
            .reset_index().drop(columns='index')
        (wd / 'plots').mkdir(parents=True, exist_ok=True)
        self.distances = dict()

        # TODO i need a background distribution: for example at every kth position?
        # TODO class balance between training and test set is really important
        # TODO use a measure that is sensitive for class imbalance, accuracy would fail horribly
        # TODO linear regression of stab change depending on pH value -> linear equation with error range
        # MARK how to deal with singleton records where no slope can be inferred? background distr

    @property
    def dataframe(self):
        return self.__dataframe__.copy(deep=True)

    def dataframe_abbrev(self, reduced=False):
        df = self.__dataframe__.copy(deep=True)
        if reduced:
            df = df.loc[df.DATASET == 'reduced_set']
        df.MUTATION = df.MUTATION.apply(abbrev)
        return df

    def dataframe_remerged(self, reduced=False, df=False):
        """Re-merge separate repeat records for different DELTAs but measured at identical pH"""
        if type(df) != pd.DataFrame:
            print('fetching new dataframe')
            df = self.dataframe
            if reduced:
                df = df.loc[df.DATASET == 'reduced_set']
            df.MUTATION = df.MUTATION.apply(abbrev)

        # # pandas DataFrame get with multiple conditions including index then assign
        # df.loc[(df.index == mo[path][0]) & (df['gene'] == gene), c2[col]] = new_text

        def remerge_records(gdf):
            for m in self.order:
                gdf[m] = gdf[m].sum(skipna=True)
            return gdf.iloc[0]

        df = df.groupby(['UniProt_ID', 'MUTATION', 'pH']) \
            .apply(remerge_records).reset_index(drop=True)

        # drop all rows that do not have at least two values
        df = df.loc[((df.h2o != 0) & ((df.ddg != 0) | (df.dtemp != 0)))  # h2o and at least one other
                    | ((df.ddg != 0) & (df.dtemp != 0)),].reset_index(drop=True)  # the other two

        df['desc'] = [' & '.join(b for b in a if b) for a in
                      zip([[None, 'ddg'][i] for i in list(df.ddg != 0)],
                          [[None, 'dtemp'][i] for i in list(df.dtemp != 0)],
                          [[None, 'h2o'][i] for i in list(df.h2o != 0)])]

        return df

    def dataframe_repeats_avg(self, reduced=False):
        """
        Use the average measured stability change, ignoring actual repeats and pH series.
        Whether a measurement is part of a series at the same pH does not matter.
        """
        df = self.dataframe
        if reduced:
            df = df.loc[df.DATASET == 'reduced_set']
        df.MUTATION = df.MUTATION.apply(abbrev)
        return df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).mean().reset_index()

    def dataframe_gaussian_avg(self, reduced=False):
        """
        Scale the measured stability change depending on the pH along a Gaussian distribution,
        where values measured farther from pH=7 are weighted less.
        """
        df = self.dataframe
        if reduced:
            df = df.loc[df.DATASET == 'reduced_set']
        df.MUTATION = df.MUTATION.apply(abbrev)

        # calculate scaling factors from the pH
        factors = df.pH.apply(norm.pdf, args=(7, 2)) / norm.pdf(7, 7, 2)
        factors.fillna(1, inplace=True)

        # scale all three measurements
        df.dtemp *= factors
        df.ddg *= factors
        df.h2o *= factors

        return df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).mean().reset_index()

    def read_mbeds(self, extend=0, h5_file=Path('.').resolve().parent
                                           / 'all_sequences_prothermdb_HALF.h5'):
        """
        Reads embeddings from the H5 file to self.mbeds.
        :param extend: The number of neighbors on each size; e.g. 8 means a region size of 17
        :param h5_file: the path to the file with embeddings
        :return: outfile: the path to the pickled extracted embeddings
        """
        print(f'reading {h5_file} ...')
        mbeds = dict()
        with h5py.File(h5_file, 'r') as f:
            for i, key in enumerate(f.keys()):
                pieces = key.split('_')
                uniprot_id, variant = pieces[0], '_'.join(pieces[1:])
                if uniprot_id not in mbeds:
                    mbeds[uniprot_id] = dict()
                if not variant:
                    mbeds[uniprot_id]['wt'] = np.array(f[key])
                else:
                    positions = [int(p[:-1]) - 1 for p in pieces[1:]]
                    # extend to neighbourhood
                    positions = sorted(set([c for ran in [list(range(
                        max(0, p - extend), min(len(f[key]), p + extend + 1)))
                        for p in positions] for c in ran]))
                    mbeds[uniprot_id][variant] = np.array(f[key])[positions, :]

        self.__mbeds__ = mbeds
        outfile = h5_file.parent / f'extracted_{extend}.pkl'

        with open(outfile, 'wb') as f:
            pickle.dump(mbeds, f)

        print(f'read {sum(len(v) for v in mbeds.values())} embeddings '
              f'for {len(mbeds)} proteins, each SAV with {extend} '
              f'neighbors on each side, wrote to {outfile}')
        return outfile

    @staticmethod
    def load_embeddings(infile):
        mbeds = dict()
        with open(infile, 'rb') as f:
            mbeds = pickle.load(f)
        return mbeds

    def fetch_df_with_pairwise_distances(self, extend=0, df=False, reduced=True,
                                         modify=None, scaler=None, func=sum):
        """
        :param extend: The number of additional neighbours to include on each side
        :param df: optional dataframe, otherwise will be dataframe_abbrev(reduced=reduced)
        :param reduced: if only the redundancy-reduced dataset shall be loaded from the df
        :param modify: 'flip' distances for negative changes, use the 'abs' value, only 'pos' or 'neg'
        :param scaler: 'std' or 'minmax'
        :param func: the function to handle compound mutations: np.mean, sum, max, min
        :return:
        """
        if not df:
            df = self.dataframe_abbrev(reduced=reduced)

        self.read_mbeds(extend=extend)
        mbeds = self.__mbeds__
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

        df = df.loc[:, self.order + pairwise_metrics].melt(
            id_vars=self.order, value_vars=pairwise_metrics,
            var_name='metric', value_name='dist').melt(
            id_vars=['metric', 'dist'], value_vars=self.order,
            var_name='delta', value_name='change')
        df = df[~df.change.isna()].reset_index(drop=True)  # drop all the NaN lines. it's ok why they were there

        if modify == 'abs':
            df.change = df.change.abs()  # make all changes positive
        elif modify == 'pos':
            df = df.loc[df.change > 0]
        elif modify == 'neg':
            df = df.loc[df.change < 0]

        df.delta = df.delta.apply(self.order.index)
        df.metric = df.metric.apply(pairwise_metrics.index)
        df = df[['delta', 'metric', 'dist', 'change']]  # re-order

        self.__dataframe_pairwise__ = df
        return mbed_dists(data=df.copy(deep=True), delta_labels=self.order, metric_labels=pairwise_metrics)
