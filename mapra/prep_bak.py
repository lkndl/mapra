import re
import warnings
from argparse import Namespace
from pathlib import Path

import pandas as pd
from Bio import SeqIO


class dataset:
    uniprot_header_rgx = re.compile(r'(?P<db>(?:sp|tr))\|(?P<accession>.+?)\|(?P<name>\S+?) ' \
                                    '(?P<full_name>.+?) OS=(?P<organism>.+?) OX=(?P<taxon_id>.+?) ' \
                                    '(GN=(?P<gene>.+?) )?PE=(?P<evidence_level>.+?) SV=(?P<version>.+?)$')
    path_regex = re.compile(r'.*?_prothermdb_(?P<measure>.+?)(?:(?:_)?(?P<dataset>rep_seq|a?)?\.fasta|\.tsv)')
    mutation_regex = re.compile(r'( ?(?:\S+:)?[ARNDCQEGHILKMFPSTWYV]\d{1,9}[ARNDCQEGHILKMFPSTWYV] ?)+')

    def __init__(self, wd=Path('.')):

        self.full_set = Namespace(**{'seq_annotations': dict()})
        self.reduced_set = Namespace(**{'seq_annotations': dict()})
        self.metrics = set()
        seq_lengths = dict()
        columns_hashes = set()

        def get_annotations(reduced=True):
            if reduced:
                return self.reduced_set.seq_annotations
            else:
                return self.full_set.seq_annotations

        for fasta in wd.rglob('*.fasta'):
            gd = dataset.path_regex.match(fasta.name).groupdict()
            if not gd:
                warnings.warn('unexpected FASTA filename: ' + str(fasta), RuntimeWarning)
                continue
            metric = gd['measure']
            self.metrics.add(metric)
            seq_annotations = get_annotations(bool(gd['dataset']))

            if metric not in seq_annotations:
                seq_annotations[metric] = dict()
            metric_annotations = seq_annotations[metric]

            for record in SeqIO.parse(fasta, 'fasta'):
                m = dataset.uniprot_header_rgx.match(record.description)
                if not m:
                    warnings.warn('unexpected FASTA header: ' + record.description, RuntimeWarning)
                    continue
                # save header information
                record_annotation = m.groupdict()
                # as well as path and sequence length
                record_annotation.update(path=str(fasta), length=len(record))
                # using the UniProt accession as id
                acc = record_annotation['accession']
                if acc in metric_annotations:
                    warnings.warn('overwriting annotations for ' + acc)
                metric_annotations[acc] = record_annotation

                # save lengths in flat dictionary
                if acc in seq_lengths and seq_lengths[acc] != len(record):
                    warnings.warn('conflicting sequence lengths for ' + acc)
                seq_lengths[acc] = len(record)

        self.tables = dict()

        for tsv in wd.rglob('*.tsv'):
            gd = dataset.path_regex.match(tsv.name).groupdict()
            if not gd:
                warnings.warn('unexpected TSV filename: ' + str(tsv), RuntimeWarning)
                continue
            metric = gd['measure']
            if metric not in self.metrics:
                warnings.warn('unexpected metric: ' + metric, RuntimeWarning)

            df = pd.read_csv(tsv, sep='\t')
            # filter out rows with undetermined '-' or 'wild-type' mutation
            df = df.loc[~df.MUTATION.isin(['-', 'wild-type'])]
            # filter out rows with undetermined UniProt_ID
            df = df.loc[df.UniProt_ID != '-']
            df[['MUTATION', 'SOURCE']] = df.MUTATION.str.rstrip(')').str.split(' \(Based on ', expand=True)
            df['MUT_COUNT'] = df.MUTATION.str.strip().str.count(' ') + 1
            df['DATASET'] = df.UniProt_ID.isin(self.reduced_set.seq_annotations[metric].keys()) \
                .astype(int).map(lambda c: ['full_set', 'reduced_set'][c])
            df.loc[~df.MUTATION.str.match(dataset.mutation_regex), 'DATASET'] = 'invalid'
            df['LENGTH'] = df.UniProt_ID.apply(lambda uniprot: seq_lengths.get(uniprot, 0))
            df['REPEATS'] = df.groupby(['UniProt_ID', 'MUTATION']).transform('count')['LENGTH']

            columns_hashes.add(hash(tuple(sorted(df.columns))))
            self.tables[metric] = df.reset_index(drop=True)

        assert len(columns_hashes) == 1, 'TSV headers were not identical'
        self.__dataframe__ = pd.concat(self.tables.values(), keys=self.tables.keys()) \
            .reset_index().rename(columns={'level_0': 'DELTA'}).drop(columns='level_1') \
            .sort_values(by=['UniProt_ID', 'DELTA']).reset_index().drop(columns='index')

    @property
    def dataframe(self):
        return self.__dataframe__.copy(deep=True)
