import re
import warnings
from pathlib import Path

import pandas as pd
from Bio import SeqIO


class dataset:
    path_regex = re.compile(r'.*?_prothermdb_(?P<measurement>.+?)(?:(?:_)?(?P<dataset>rep_seq|a?)?\.fasta|\.tsv)')
    mutation_regex = re.compile(r'( ?(?:\S+:)?[ARNDCQEGHILKMFPSTWYV]\d{1,9}[ARNDCQEGHILKMFPSTWYV] ?)+')

    def __init__(self, wd=Path('.')):

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
        df['REPEATS'] = df.groupby(['UniProt_ID', 'MUTATION', 'DELTA']).transform('count')['LENGTH']

        self.__dataframe__ = df.sort_values(by=['UniProt_ID', 'MUTATION']) \
            .reset_index().drop(columns='index')
        self.full_set_lengths, self.reduced_set_lengths = sets

    @property
    def dataframe(self):
        return self.__dataframe__.copy(deep=True)
