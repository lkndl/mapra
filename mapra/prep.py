import pickle
import re
import warnings
import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.stats
from Bio import SeqIO
from scipy.stats import norm
from sklearn.metrics.pairwise import paired_distances


def _abbrev(mutation_pattern):
    return '_'.join(i[1:] for i in mutation_pattern.split(' ') if i)


def save(fig, path):
    """Save a matplotlib or seaborn figure in the plots subdirectory with a given path/name."""
    wd = Path('.').resolve().parent / 'plots'
    Path.mkdir((wd / path).parent, parents=True, exist_ok=True)
    fig.savefig(wd / path, dpi=300, bbox_inches='tight', pad_inches=.1)


class dataset:
    path_regex = re.compile(r'.*?_prothermdb_(?P<measurement>.+?)(?:(?:_)?(?P<dataset>rep_seq|a?)?\.fasta|\.tsv)')
    mutation_regex = re.compile(r'( ?(?:\S+:)?[ARNDCQEGHILKMFPSTWYV]\d{1,9}[ARNDCQEGHILKMFPSTWYV] ?)+')

    def __init__(self, wd=Path('.').resolve().parent, legacy=False):
        print(wd)
        sets = [dict(), dict()]  # full_set, reduced_set
        self.__mbeds__ = None
        self.__dataframe_pairwise__ = None
        self.__library__ = dict()
        self.__rng__ = np.random.default_rng(12345)

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
            warnings.warn('found not exactly one TSV with annotations:\n' + '\n'.join(str(tsv) for tsv in tsvs))

        protherm = [tsv for tsv in tsvs if tsv.stem == 'prothermdb_annotations'][0]
        df = pd.read_csv(protherm, sep='\t')

        # read PDB IDs from the other tsv
        pdb_ids = [tsv for tsv in tsvs if tsv.stem == 'uniprot_to_pdb'][0]
        pdb = dict()
        for row in pd.read_csv(pdb_ids, sep='\t', header=0).itertuples():
            pdb[row.From] = row.To
        df['PDB'] = df.UniProt_ID.apply(pdb.get)

        # # replace pH column with ∆pH using other tsv with average wildtype pH values
        # pH_tsv = [tsv for tsv in tsvs if tsv.stem == 'wildtype_pHs'][0]
        # pHs = dict()
        # for row in pd.read_csv(pH_tsv, sep='\t', header=0).itertuples():
        #     pHs[row.wildtype] = row.pH
        # df['wildtype_pH'] = df.UniProt_ID.apply(pHs.get)
        # df['delta_pH'] = df.pH - df.wildtype_pH
        # # df.update(df['delta_pH'].fillna(0))
        # # df = df.drop(columns=['pH', 'wildtype_pH'])

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

    @property
    def dataframe(self):
        return self.__dataframe__.copy(deep=True)

    def dataframe_abbrev(self, reduced=False):
        df = self.__dataframe__.copy(deep=True)
        if reduced:
            df = df.loc[df.DATASET == 'reduced_set']
        df.MUTATION = df.MUTATION.apply(_abbrev)
        return df

    def dataframe_remerged(self, reduced=False, df=False):
        """Re-merge separate repeat records for different DELTAs but measured at identical pH"""
        if type(df) != pd.DataFrame:
            print('fetching new dataframe')
            df = self.dataframe
            if reduced:
                df = df.loc[df.DATASET == 'reduced_set']
            df.MUTATION = df.MUTATION.apply(_abbrev)

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
        df.MUTATION = df.MUTATION.apply(_abbrev)
        return df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).mean().reset_index()

    def dataframe_gaussian_avg(self, reduced=False):
        """
        Scale the measured stability change depending on the pH along a Gaussian distribution,
        where values measured farther from pH=7 are weighted less.
        """
        df = self.dataframe
        if reduced:
            df = df.loc[df.DATASET == 'reduced_set']
        df.MUTATION = df.MUTATION.apply(_abbrev)

        # calculate scaling factors from the pH
        factors = df.pH.apply(norm.pdf, args=(7, 2)) / norm.pdf(7, 7, 2)
        factors.fillna(1, inplace=True)

        # scale all three measurements
        df.dtemp *= factors
        df.ddg *= factors
        df.h2o *= factors

        return df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).mean().reset_index()

    def _extract_embeds(self, extend=0, h5_file=Path('.').resolve().parent
                                                / 'all_sequences_prothermdb_HALF.h5'):
        """
        Extracts embeddings from a H5 file to .pkl file, saves the paths in
        self.__library__ and returns the dict of embeddings
        :param extend: The number of neighbors on each size; e.g. 8 means a region size of 17
        :param h5_file: the path to the file with embeddings
        :return: mbeds: the extracted embeddings as a dictionary
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

        outfile = h5_file.parent / 'pkl' / f'h5_slice_{extend}.pkl'
        self.__library__[extend] = outfile
        with open(outfile, 'wb') as f:
            pickle.dump(mbeds, f)

        print(f'read {sum(len(v) for v in mbeds.values())} embeddings '
              f'for {len(mbeds)} proteins, each SAV with {extend} '
              f'neighbors on each side, wrote to {outfile}')
        return mbeds

    def fetch_mbeds(self, extend=0):
        """
        Fetch embeddings depending on the number of
        additional positions each side of a mutation.
        """
        # TODO cheating:
        self.__library__[0] = Path('/home/quirin/PYTHON/mapra/pkl/h5_slice_0.pkl')

        if extend in self.__library__:
            with open(self.__library__[extend], 'rb') as f:
                print(f'loading from {self.__library__[extend]}')
                return pickle.load(f)
        else:
            return self._extract_embeds(extend=extend)

    def fetch_numpy_distances(self, df=None, reduced=True, test_sets=None):
        """
        For a given pandas DataFrame (or the default df with abbreviated mutation patterns),
        build a numpy array mapping all three metrics of protein stability change to the
        changes in the embeddings; saving all 1024 dimensions.
        :param df: a ProThermDB pandas DataFrame, otherwise self.dataframe_abbrev(reduced=reduced)
        :param reduced: bool, use the redundancy-reduced dataset or not
        :param test_set: if a list is passed, the additional new first column is 0 for train rows and 1 for test rows
        :return:
        """
        if test_sets is None:
            test_sets = {delta: list() for delta in self.order}  # make an empty dummy so lookup works
        if df is None:
            df = self.dataframe_abbrev(reduced=reduced)
        mbeds = self.fetch_mbeds(0)

        # iterate over the embeddings once, then access the smaller
        # result multiple times to build an array matching the dataframe
        npdists = dict()
        for uniprot_id, d in mbeds.items():
            wt = d.pop('wt')
            try:
                # make sure that even if something goes wrong, we put back the wildtype
                npdists[uniprot_id] = dict()
                # the variants already come with their array slices of the right size
                for variant, ar in d.items():
                    positions = [int(p[:-1]) - 1 for p in variant.split('_')]
                    # for each mutated position, subtract the wildtype
                    # from the variant and add up these rows of differences
                    npdists[uniprot_id][variant] = np.sum(np.subtract(ar, wt[positions, :]),
                                                          axis=0, keepdims=True)
            except Exception as ex:
                raise RuntimeError(f'Something failed for {uniprot_id}') from ex
            d['wt'] = wt

        # do some pandas magic and a numpy array pops out
        npr = np.vstack(df.apply(
            lambda gdf: np.hstack((  # glue to the left side of the row of differences
                np.array([[int(gdf.UniProt_ID in test_sets[gdf.DELTA]),  # 0/1 if the row belongs to this train/test set
                           self.order.index(gdf.DELTA),  # which metric was measured
                           gdf[gdf.DELTA]]], dtype=np.float16),  # and the measured value
                npdists.get(gdf.UniProt_ID, dict()).get(
                    gdf.MUTATION, np.zeros((1, 1024), dtype=np.float16))  # fall back to zeroes is needed
            )), axis=1))

        if not test_sets:
            return npr[:, 1:]  # cut off the train/test column
        return npr

    def fetch_df_with_pairwise_distances(self, extend=0, df=None, reduced=True,
                                         modify=None, scaler=None, func=sum, epsilon=0):
        """
        :param extend: The number of additional neighbours to include on each side
        :param df: optional dataframe, otherwise will be dataframe_abbrev(reduced=reduced)
        :param reduced: if only the redundancy-reduced dataset shall be loaded from the df
        :param modify: 'flip' distances for negative changes, use the 'abs' value, only 'pos' or 'neg'
        :param scaler: 'std' or 'minmax'
        :param func: the function to handle compound mutations: np.mean, np.prod, sum, max, min
        :param epsilon: a constant that will be added to allow meaningful products
        :return:
        """
        if df is None:
            df = self.dataframe_abbrev(reduced=reduced)
        mbeds = self.fetch_mbeds(extend)

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
                        epsilon + paired_distances(wt[positions, :],
                                                   ar, metric=m)) for m in pairwise_metrics}
            except Exception as ex:
                raise RuntimeError(f'Something failed for {uniprot_id}') from ex
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
        df = df[~df.change.isna()].reset_index(drop=True)
        # drop all the NaN lines. it's ok why they were there

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

        @dataclasses.dataclass
        class mbed_dists:
            data: pd.DataFrame
            delta_labels: list
            metric_labels: list

        return mbed_dists(data=df.copy(deep=True), delta_labels=self.order, metric_labels=pairwise_metrics)

    def fetch_spearman_rhos(self, npr):
        spears = dict()
        for i, delta in enumerate(self.order):
            # select only matching rows, and ignore the delta column
            dnpr = npr[npr[:, 0] == i, 1:]
            spear, pval = scipy.stats.spearmanr(dnpr, axis=0)
            # only need first line[1:] of output matrix
            spears[delta] = spear[0, 1:]
        return spears

    def uniprot_train_test_split(self, df=None, reduced=True, test_size=.2, random_state=None):
        """
        Splits a ProThermDB dataset into a training and final testing set along
        UniProt IDs using a given target test size that will be approximately true.

        :param df: a ProThermDB pandas DataFrame, otherwise self.dataframe_abbrev(reduced=reduced)
        :param reduced: bool, use the redundancy-reduced dataset or not
        :param test_size: the target test size, will only be approximately true
        :param random_state: seed that determines the random selection of the test set
        :return: Return a dict with delta labels as keys containing a dataclass
        instance similar to sklearn.model_selection.train_test_split.
        """
        assert 0 < test_size < 1, 'invalid test size, must be 0 < test_size < 1'

        if df is None:
            df = self.dataframe_abbrev(reduced=reduced)
        if random_state is None:
            random_state = self.__rng__.integers(low=0, high=1000, size=1)[0]
        local_rng = np.random.default_rng(random_state)

        @dataclasses.dataclass
        class Split:
            delta: str
            X: np.array = None
            X_test: np.array = None
            y: np.array = None
            y_true: np.array = None
            test_set: set = None
            real_test_size: float = None
            records_test_size: float = None

        splits = dict()
        test_sets = dict()
        for i, delta in enumerate(self.order):
            # get ids for this metric as a sorted list
            uniprot_ids = sorted(set(df.loc[df.DELTA == delta, 'UniProt_ID']))
            # calculate how many distinct UniProt IDs will be in the test set
            abs_test_size = int(np.ceil(len(uniprot_ids) * test_size))
            assert abs_test_size < len(uniprot_ids), 'no training data left, set smaller test size'
            # shuffle the sorted list of all UniProt IDs
            local_rng.shuffle(uniprot_ids)
            # use leading entries as test set
            test_set = uniprot_ids[:abs_test_size]
            # save for fetch_numpy_distances
            test_sets[delta] = test_set
            # make and pre-fill the Split dataclass instance
            splits[delta] = Split(delta=delta, test_set=set(test_set),
                                  real_test_size=len(test_set) / len(uniprot_ids))

        # get embedding changes
        npr = self.fetch_numpy_distances(df=df, test_sets=test_sets)
        train_filter = npr[:, 0] == 0
        train_npr, test_npr = npr[train_filter, 1:], npr[~train_filter, 1:]

        def get_features_and_labels_for_delta(ar, i):
            # select the rows for this delta, and cleave off the delta column
            dar = ar[ar[:, 0] == i, 1:]
            # split into features and labels
            return dar[:, 1:], dar[:, 0].reshape(-1, 1)

        # records_test_sizes = dict()
        for i, delta in enumerate(self.order):
            s = splits[delta]
            s.X, s.y = get_features_and_labels_for_delta(train_npr, i)
            s.X_test, s.y_true = get_features_and_labels_for_delta(test_npr, i)
            s.records_test_size = len(s.y_true) / (len(s.y) + len(s.y_true))

        pp = lambda it: ':'.join('%.4f' % elem.__getattribute__(it) for elem in splits.values())
        print(f'split {random_state} targeted {test_size}, '
              f'real test sizes: {pp("real_test_size")}, record test sizes: {pp("records_test_size")}')
        return splits
