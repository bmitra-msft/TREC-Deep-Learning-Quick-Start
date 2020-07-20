import os
import csv
import random
import torch
from torch.utils import data
from collections import Counter
import numpy as np


class TRECSupervisedTrainMultiDataset(data.Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.max_dataset_len = max([len(dataset) for dataset in self.datasets])

    def __len__(self):
        return self.max_dataset_len * self.num_datasets

    def __getitem__(self, idx):
        dataset = self.datasets[idx // self.max_dataset_len]
        idx = idx % self.max_dataset_len
        idx = idx % len(dataset)
        return dataset[idx]


class TRECSupervisedTrainDataset(data.Dataset):

    def __init__(self, query_set, featurize, utils):
        self.featurize = featurize
        self.utils = utils
        self.f_docs = open(os.path.join(utils.args.local_dir, utils.args.file_in_docs), 'rt', encoding='utf8')
        self.f_orcas = open(os.path.join(utils.args.local_dir, utils.args.file_gen_orcas_docs), 'rt', encoding='utf8')
        self.qids = getattr(utils, 'qids_{}'.format(query_set))
        self.cand = getattr(utils, 'cand_{}'.format(query_set))
        self.qrels = getattr(utils, 'qrels_{}'.format(query_set))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        q = self.utils.qs[qid]
        dids = self.cand[qid]
        qrels = self.qrels.get(qid, {})
        labeled = {}
        for did in dids:
            label = qrels.get(did, 0) + 1
            if label not in labeled:
                labeled[label] = []
            labeled[label].append(did)
        if len(labeled) > 1:
            sampled_labels = sorted(random.sample(labeled.keys(), 2), reverse=True)
        else:
            sampled_labels = list(labeled.keys())
        cands = [(random.sample(labeled[label], 1)[0], label) for label in sampled_labels]
        num_rand_negs = self.utils.args.num_rand_negs + 2 - len(cands)
        if num_rand_negs > 0:
            cands += [(did, 0) for did in random.sample(self.utils.dids, num_rand_negs)]
        dids = [x[0] for x in cands]
        ds = [self.utils.get_doc_content(self.f_docs, self.f_orcas, did) for did in dids]
        labels = torch.FloatTensor(np.asarray([x[1] for x in cands], dtype=np.float32))
        features = self.featurize(q, ds)
        return (qid, dids, labels, features)


class TRECSupervisedTestDataset(data.Dataset):

    def __init__(self, query_set, featurize, utils):
        self.featurize = featurize
        self.utils = utils
        self.cand = getattr(utils, 'cand_{}'.format(query_set))
        self.cand = [[(qid, did) for did in cands] for qid,cands in self.cand.items()]
        self.cand = [item for sublist in self.cand for item in sublist]
        self.qrels = getattr(utils, 'qrels_{}'.format(query_set)) if hasattr(utils, 'qrels_{}'.format(query_set)) else {}
        self.f_docs = open(os.path.join(utils.args.local_dir, utils.args.file_in_docs), 'rt', encoding='utf8')
        self.f_orcas = open(os.path.join(utils.args.local_dir, utils.args.file_gen_orcas_docs), 'rt', encoding='utf8')

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):
        qid, did = self.cand[idx]
        qrels = self.qrels.get(qid, {})
        label = qrels.get(did, 0)
        cand = [(did, label)]
        labels = torch.FloatTensor(np.asarray([label], dtype=np.float32))
        features = self.featurize(self.utils.qs[qid], [self.utils.get_doc_content(self.f_docs, self.f_orcas, did)])
        return (qid, did, labels, features)


class TRECInferenceDataset(data.IterableDataset):

    def __init__(self, query_set, featurize, tokenize, utils):
        self.featurize = featurize
        self.utils = utils
        self.qids = getattr(utils, 'qids_{}'.format(query_set))
        self.qid_to_terms = {qid: tokenize(utils.qs[qid])[:utils.args.max_terms_query] for qid in self.qids}
        self.query_terms = list(set([item for sublist in self.qid_to_terms.values() for item in sublist]))
        self.query = ' '.join(self.query_terms)
        self.f_docs = open(os.path.join(utils.args.local_dir, utils.args.file_in_docs), 'rt', encoding='utf8')
        self.f_orcas = open(os.path.join(utils.args.local_dir, utils.args.file_gen_orcas_docs), 'rt', encoding='utf8')

    def __iter__(self):
        for did in self.utils.dids:
            features = self.featurize(self.query, [self.utils.get_doc_content(self.f_docs, self.f_orcas, did)], infer_mode=True)
            yield (did, features)
