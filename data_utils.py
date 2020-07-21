import os
import csv
import sys
import gzip
import random
import shutil
import tarfile
import requests
import numpy as np
from clint.textui import progress
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError


class DataUtils:

    def __init__(self, printer):
        self.printer = printer
        self.prenorm_file_cols = {'file_in_docs': [1, 2, 3], 'file_in_orcas': [1], 'file_in_qs_train': [1], 'file_in_qs_dev': [1], 'file_in_qs_val': [1], 'file_in_qs_test': [1], 'file_in_qs_orcas': [1]}
        csv.field_size_limit(sys.maxsize)

    def parser_add_args(self, parser):
        parser.add_argument('--local_dir', default='data/', help='root directory for data files (default: data/)')
        parser.add_argument('--web_dir', default='https://msmarco.blob.core.windows.net/msmarcoranking/', help='root directory for data files (default: /data/home/bmitra/data/trec2019-doc/)')
        parser.add_argument('--file_in_docs', default='msmarco-docs.tsv', help='filename for document collection (default: msmarco-docs.tsv)')
        parser.add_argument('--file_in_orcas', default='orcas.tsv', help='filename for orcas data (default: orcas.tsv)')
        parser.add_argument('--file_in_qs_train', default='msmarco-doctrain-queries.tsv', help='filename for train queries (default: msmarco-doctrain-queries.tsv)')
        parser.add_argument('--file_in_qs_dev', default='msmarco-docdev-queries.tsv', help='filename for development queries (default: msmarco-docdev-queries.tsv)')
        parser.add_argument('--file_in_qs_val', default='msmarco-test2019-queries.tsv', help='filename for validation queries (default: msmarco-test2019-queries.tsv)')
        parser.add_argument('--file_in_qs_test', default='msmarco-test2020-queries.tsv', help='filename for test queries (default: msmarco-test2020-queries.tsv)')
        parser.add_argument('--file_in_qs_orcas', default='orcas-doctrain-queries.tsv', help='filename for orcas queries (default: orcas-doctrain-queries.tsv)')
        parser.add_argument('--file_in_cnd_train', default='msmarco-doctrain-top100', help='filename for top 100 train candidates (default: msmarco-doctrain-top100)')
        parser.add_argument('--file_in_cnd_dev', default='msmarco-docdev-top100', help='filename for top 100 dev candidates (default: msmarco-docdev-top100)')
        parser.add_argument('--file_in_cnd_val', default='msmarco-doctest2019-top100', help='filename for top 100 validation candidates (default: msmarco-doctest2019-top100)')
        parser.add_argument('--file_in_cnd_test', default='msmarco-doctest2020-top100', help='filename for top 100 test candidates (default: msmarco-doctest2020-top100)')
        parser.add_argument('--file_in_cnd_orcas', default='orcas-doctrain-top100', help='filename for orcas candidates (default: orcas-doctrain-top100)')
        parser.add_argument('--file_in_qrel_train', default='msmarco-doctrain-qrels.tsv', help='filename for train qrels (default: msmarco-doctrain-qrels.tsv)')
        parser.add_argument('--file_in_qrel_dev', default='msmarco-docdev-qrels.tsv', help='filename for dev qrels (default: msmarco-docdev-qrels.tsv)')
        parser.add_argument('--file_in_qrel_val', default='2019qrels-docs.txt', help='filename for validation qrels (default: 2019qrels-docs.txt)')
        parser.add_argument('--file_in_qrel_orcas', default='orcas-doctrain-qrels.tsv', help='filename for orcas qrels (default: orcas-doctrain-qrels.tsv)')
        parser.add_argument('--file_gen_docs_lookup', default='lookup-docs-norm.tsv', help='filename for document offsets for collection (default: lookup-docs-norm.tsv)')
        parser.add_argument('--file_gen_orcas_docs', default='orcas-docs.tsv', help='filename for orcas field (default: orcas-docs.tsv)')
        parser.add_argument('--file_gen_orcas_docs_lookup', default='lookup-orcas-docs-norm.tsv', help='filename for document offsets for orcas field (default: lookup-docs-orcas-norm.tsv)')
        parser.add_argument('--num_fields', default=4, help='number of fields per document (default: 4)', type=int)
        parser.add_argument('--num_dev_queries', default=100, help='number of queries to sample for dev set (default: 100)', type=int)

    def parser_validate_args(self, args):
        self.args = args
        if not os.path.exists(args.local_dir):
            os.makedirs(args.local_dir)

    def setup_and_verify(self):
        self.__verify_in_data()
        self.__verify_gen_data()
        self.__preload_data_to_memory()

    def get_doc_content(self, f_docs, f_orcas, did):
        if did == '':
            return [''] * self.args.num_fields
        f_docs.seek(self.doc_offsets[did])
        line = f_docs.readline()
        assert line.startswith(did + "\t"), 'looking for {} at position {}, found {}'.format(did, self.doc_offsets[did], line)
        field_values = line.split('\t')[1:]
        if did in self.orcas_docs_offsets:
            f_orcas.seek(self.orcas_docs_offsets[did])
            line = f_orcas.readline()
            assert line.startswith(did + "\t"), 'looking for {} at position {}, found {}'.format(did, self.orcas_docs_offsets[did], line)
            orcas_field = line.split('\t')[1]
        else:
            orcas_field = ''
        field_values.append(orcas_field)
        return field_values

    def __preload_data_to_memory(self):
        self.printer.print('preloading data to memory')
        self.doc_offsets = self.__get_doc_offsets(os.path.join(self.args.local_dir, self.args.file_gen_docs_lookup))
        self.orcas_docs_offsets = self.__get_doc_offsets(os.path.join(self.args.local_dir, self.args.file_gen_orcas_docs_lookup))
        self.dids = list(self.doc_offsets.keys())
        qs_train = self.__load_set('train')
        qs_dev = self.__load_set('dev', num_samples=self.args.num_dev_queries)
        qs_val = self.__load_set('val')
        qs_test = self.__load_set('test')
        if self.args.orcas_train:
            qs_orcas = self.__load_set('orcas')
            self.qs = {**qs_train, **qs_orcas, **qs_dev, **qs_val, **qs_test}
        else:
            self.qs = {**qs_train, **qs_dev, **qs_val, **qs_test}
        setattr(self.args, 'collection_size', len(self.doc_offsets))
        setattr(self.args, 'num_train_queries', len(qs_train))
        self.args.num_dev_queries = len(qs_dev)

    def __verify_in_data(self):
        self.printer.print('verifying input data')
        for k, file_name in vars(self.args).items():
            if k.startswith('file_in_'):
                expect_prenorm = self.__should_prenorm_file(k)
                if expect_prenorm:
                    file_norm = self.__get_post_norm_filename(file_name)
                    if self.__verify_and_download_file(file_norm):
                        setattr(self.args, k, file_norm)
                        continue
                if self.__verify_and_download_file(file_name):
                    if expect_prenorm:
                        self.__prenorm_input_file(k, os.path.join(self.args.local_dir, file_name), os.path.join(self.args.local_dir, file_norm))
                        setattr(self.args, k, file_norm)
                else:
                    self.printer.print('error: can not find file {}'.format(file_name))
                    sys.exit(0)

    def __verify_gen_data(self):
        self.printer.print('verifying intermediate data')
        for k, file_name in vars(self.args).items():
            if k.startswith('file_gen_'):
                if not self.__verify_and_download_file(file_name):
                    if k == 'file_gen_docs_lookup':
                        self.__generate_lookup()
                    elif k == 'file_gen_orcas_docs' or k == 'file_gen_orcas_docs_lookup':
                        self.__generate_orcas_field()

    def __should_prenorm_file(self, file_key):
        return (file_key in self.prenorm_file_cols)

    def __get_post_norm_filename(self, file_name):
        return file_name + '.norm'

    def __prenorm_input_file(self, file_key, file_path, file_path_norm):
        self.printer.print('normalizing {}'.format(file_path))
        with open(file_path, 'rt', encoding='utf8') as f_in:
            with open(file_path_norm, 'w', encoding='utf8') as f_out:
                reader = csv.reader(f_in, delimiter='\t', quoting=csv.QUOTE_NONE)
                cols_to_clean = self.prenorm_file_cols[file_key]
                for row in reader:
                    clean_cols = []
                    for i in range(len(row)):
                        clean_cols.append(self.parent.model_utils.clean_text(row[i]) if i in cols_to_clean else row[i])
                    clean_text = '\t'.join(clean_cols)
                    f_out.write(clean_text)
                    f_out.write('\n')
        os.remove(file_path)

    def __generate_lookup(self):
        self.printer.print('generating document offsets for collection')
        with open(os.path.join(self.args.local_dir, self.args.file_in_docs), 'rt', encoding='utf8') as f_in:
            with open(os.path.join(self.args.local_dir, self.args.file_gen_docs_lookup), 'w', encoding='utf8') as f_out:
                offset = 0
                line = f_in.readline()
                while line:
                    did = line.split('\t')[0]
                    f_out.write('{}\t{}\n'.format(did, offset))
                    offset = f_in.tell()
                    line = f_in.readline()

    def __generate_orcas_field(self):
        self.printer.print('generating orcas field data')
        orcas_field = {}
        with open(os.path.join(self.args.local_dir, self.args.file_in_orcas), 'rt', encoding='utf8') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for [qid, q, did, _] in reader:
                if did not in orcas_field:
                    orcas_field[did] = []
                orcas_field[did].append(q)
        orcas_field = {k: ' '.join(v) for k,v in orcas_field.items()}
        with open(os.path.join(self.args.local_dir, self.args.file_gen_orcas_docs), 'w', encoding='utf8') as f_out:
            with open(os.path.join(self.args.local_dir, self.args.file_gen_orcas_docs_lookup), 'w', encoding='utf8') as f_lookup:
                offset = 0
                for did, field in orcas_field.items():
                    f_out.write('{}\t{}\n'.format(did, field))
                    f_lookup.write('{}\t{}\n'.format(did, offset))
                    offset = f_out.tell()

    def __get_doc_offsets(self, lookup_file):
        offsets = {}
        with open(lookup_file, 'rt', encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            for [did, offset] in reader:
                offsets[did] = int(offset)
        return offsets

    def __load_set(self, query_set, num_samples=0):
        file_in_qs = getattr(self.args, 'file_in_qs_{}'.format(query_set))
        qs = self.__get_qs(file_in_qs)
        file_in_cnd = getattr(self.args, 'file_in_cnd_{}'.format(query_set))
        cand = self.__get_candidates(file_in_cnd)
        qids = set(qs.keys()) & set(cand.keys())
        if query_set != 'test':
            file_in_qrel = getattr(self.args, 'file_in_qrel_{}'.format(query_set))
            qrels = self.__get_qrels(file_in_qrel)
            qids = qids  & set(qrels.keys())
        qids = list(qids)
        if num_samples > 0:
            qids = random.sample(qids, min(num_samples, len(qids)))
        setattr(self, 'qids_{}'.format(query_set), qids)
        if query_set != 'test':
            qrels = {qid: qrels[qid] for qid in qids}
            setattr(self, 'qrels_{}'.format(query_set), qrels)
        cand = {qid: cand[qid] for qid in qids}
        setattr(self, 'cand_{}'.format(query_set), cand)
        qs = {qid: qs[qid] for qid in qids}
        return qs

    def __get_qrels(self, qrels_file):
        qrels = {}
        with open(os.path.join(self.args.local_dir, qrels_file), 'rt', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=' ')
            for [qid, _, did, rating] in reader:
                rating = int(rating)
                if rating == 0:
                    continue
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = rating
        return qrels

    def __get_candidates(self, cnd_file):
        cands = {}
        with open(os.path.join(self.args.local_dir, cnd_file), 'rt', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=' ')
            for [qid, _, did, _, _, _] in reader:
                if qid not in cands:
                    cands[qid] = [did]
                else:
                    cands[qid].append(did)
        return cands

    def __get_qs(self, qs_file):
        qs = {}
        with open(os.path.join(self.args.local_dir, qs_file), 'rt', encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            for [qid, q_txt] in reader:
                qs[qid] = q_txt
        return qs

    def __verify_and_download_file(self, file_name):
        file_local = os.path.join(self.args.local_dir, file_name)
        if not os.path.exists(file_local):
            file_local_tar = '{}.tar'.format(file_local)
            file_local_gz = '{}.gz'.format(file_local)
            file_local_tar_gz = '{}.tar.gz'.format(file_local)
            if os.path.exists(file_local_tar):
                self.__untar(file_local_tar)
            elif os.path.exists(file_local_gz):
                self.__uncompress(file_local_gz)
            elif os.path.exists(file_local_tar_gz):
                self.__untar(file_local_tar_gz)
            else:
                file_web = os.path.join(self.args.web_dir, file_name)
                file_web_tar = '{}.tar'.format(file_web)
                file_web_gz = '{}.gz'.format(file_web)
                file_web_tar_gz = '{}.tar.gz'.format(file_web)
                if self.__web_file_exists(file_web):
                    self.__download_file(file_web, file_local)
                elif self.__web_file_exists(file_web_tar):
                    self.__download_file(file_web_tar, file_local_tar)
                    self.__untar(file_local_tar)
                elif self.__web_file_exists(file_web_gz):
                    self.__download_file(file_web_gz, file_local_gz)
                    self.__uncompress(file_local_gz)
                elif self.__web_file_exists(file_web_tar_gz):
                    self.__download_file(file_web_tar_gz, file_local_tar_gz)
                    self.__untar(file_local_tar_gz)
                else:
                    return False
        return True

    def __untar(self, filename):
        self.printer.print('unpacking {}'.format(filename))
        f = tarfile.open(filename)
        f.extractall(path=os.path.dirname(filename))
        f.close()
        os.remove(filename)

    def __uncompress(self, filename, block_size=65536):
        self.printer.print('uncompressing {}'.format(filename))
        with gzip.open(filename, 'rb') as s_file:
            with open(filename[:-3], 'wb') as d_file:
                shutil.copyfileobj(s_file, d_file, block_size)
        os.remove(filename)

    def __web_file_exists(self, url):
        return requests.head(url).status_code != 404

    def __download_file(self, filename_web, filename_local):
        self.printer.print('downloading {}'.format(filename_web))
        chunk_size = 1048576
        adapter = HTTPAdapter(max_retries=10)
        session = requests.Session()
        session.mount(filename_web, adapter)
        try:
            r = session.get(filename_web, stream=True, timeout=5)
            with open(filename_local, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for ch in progress.bar(r.iter_content(chunk_size=chunk_size), expected_size=(total_length / chunk_size) + 1):
                    if ch:
                        f.write(ch)
        except ConnectionError as ce:
            self.printer.print('error: {}'.format(ce))
