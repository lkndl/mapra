import re
import sys
import warnings
from pathlib import Path
from scipy.stats import norm
import os

import pandas as pd
from Bio import SeqIO


def abbrev(mutation_pattern):
    return '_'.join(i[1:] for i in mutation_pattern.split(' '))


class dataset:
    path_regex = re.compile(r'.*?_prothermdb_(?P<measurement>.+?)(?:(?:_)?(?P<dataset>rep_seq|a?)?\.fasta|\.tsv)')
    mutation_regex = re.compile(r'( ?(?:\S+:)?[ARNDCQEGHILKMFPSTWYV]\d{1,9}[ARNDCQEGHILKMFPSTWYV] ?)+')

    def __init__(self, wd=Path('.').resolve().parent, legacy=False):
        print(wd)
        sets = [dict(), dict()]  # full_set, reduced_set

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

        # TODO i need a background distribution: for example at every kth position?
        # TODO class balance between training and test set is really important
        # TODO use a measure that is sensitive for class imbalance, accuracy would fail horribly
        # TODO linear regression of stab change depending on pH value -> linear equation with error range
        # MARK how to deal with singleton records where no slope can be inferred? background distr

    @property
    def dataframe(self):
        return self.__dataframe__.copy(deep=True)

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
        """Use the average measured stability change, ignoring actual repeats and pH series."""
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

        # scale all three measurements
        df.dtemp *= factors
        df.ddg *= factors
        df.h2o *= factors

        return df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).mean().reset_index()
