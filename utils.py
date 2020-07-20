import os
import sys
import csv
import math
import torch
import argparse
import datetime
from factory import Factory
from data_utils import DataUtils
from learner_utils import LearnerUtils


class Printer:

    def __init__(self, file_path):
        self.log = open(file_path, mode='w', encoding='utf-8')

    def print(self, s, end='\n', suppress_timestamp=False):
        if not suppress_timestamp:
            msg = '[{}]\t{}{}'.format(datetime.datetime.now().strftime('%b %d, %H:%M:%S'), s, end)
        else:
            msg = '\t{}{}'.format(s, end)
        print(msg, flush=True, end='')
        self.log.write(msg)
        self.log.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.close()


class Utils:

    def __init__(self, printer=None):
        if printer == None:
            self.printer = Printer('log.txt')
        else:
            self.printer = printer

    def setup_and_verify(self):
        parser = argparse.ArgumentParser(description= 'trec 2019 deep learning track (document re-ranking task)')
        torch.set_printoptions(threshold=500)
        self.__parser_add_args(parser)
        self.args, _ = parser.parse_known_args()
        self.data_utils = DataUtils(self.printer)
        self.learner_utils = LearnerUtils(self.printer)
        self.model_utils = Factory.get_model_utils(self)
        self.sub_utils = [self.data_utils, self.model_utils, self.learner_utils]
        for sub_utils in self.sub_utils:
            sub_utils.parent = self
            sub_utils.parser_add_args(parser)
        self.args = parser.parse_args()
        for sub_utils in self.sub_utils:
            sub_utils.parser_validate_args(self.args)
        self.__print_versions()
        for sub_utils in self.sub_utils:
            sub_utils.setup_and_verify()
        self.__print_args()

    def evaluate_baseline(self):
        results_dev = self.get_baseline_results(self.args.file_in_cnd_dev)
        mrr_dev, _, _ = self.evaluate_results(results_dev, self.data_utils.qrels_dev)
        results_val = self.get_baseline_results(self.args.file_in_cnd_val)
        mrr_val, ncg_val, ndcg_val = self.evaluate_results(results_val, self.data_utils.qrels_val)
        self.printer.print('baseline\tdev mrr: {:.3f}\tval mrr: {:.3f}\tval ncg: {:.3f}\tval ndcg: {:.3f}'.format(mrr_dev, mrr_val, ncg_val, ndcg_val))

    def get_baseline_results(self, cnd_file):
        results = {}
        with open(os.path.join(self.args.local_dir, cnd_file), 'rt', encoding='utf8') as f:
            reader = csv.reader(f, delimiter=' ')
            for [qid, _, did, rank, _, _] in reader:
                rank = int(rank)
                if qid not in results:
                    results[qid] = []
                results[qid].append((did, -rank))
        results = {qid: sorted(docs, key=lambda x: x[1], reverse=True)[:self.args.max_metric_pos_nodisc] for qid, docs in results.items()}
        return results

    def evaluate_results(self, results, qrels):
        mrr = 0
        ncg = 0
        ndcg = 0
        for qid, docs in results.items():
            if qid not in qrels:
                continue
            qrels_q = qrels[qid]
            gains = [qrels_q.get(doc[0], 0) for doc in docs]
            ideal_gains = sorted(list(qrels_q.values()), reverse=True)[:self.args.max_metric_pos_nodisc]
            max_metric_pos_disc = min(len(gains), self.args.max_metric_pos)
            max_metric_pos_disc_ideal = min(len(ideal_gains), self.args.max_metric_pos)
            cg = sum([gain for gain in gains])
            dcg = sum([gains[i] / math.log2(i + 2) for i in range(max_metric_pos_disc)])
            ideal_cg = sum([ideal_gain for ideal_gain in ideal_gains])
            ideal_dcg = sum([ideal_gains[i] / math.log2(i + 2) for i in range(max_metric_pos_disc_ideal)])
            ncg += cg / ideal_cg if ideal_cg > 0 else 0
            ndcg += dcg / ideal_dcg if ideal_dcg > 0 else 0
            try:
                mrr += 1 / ([min(gain, 1) for gain in gains][:max_metric_pos_disc].index(1) + 1)
            except Exception: 
                pass
        mrr /= len(qrels)
        ncg /= len(qrels)
        ndcg /= len(qrels)
        return mrr, ncg, ndcg

    def __parser_add_args(self, parser):
        parser.add_argument('--model', default='ndrm3', help='model architecture (default: ndrm3)')

    def __print_args(self):
        self.printer.print('Running with following specified and inferred arguments:')
        for key, value in self.args._get_kwargs():
            if value is not None:
                if isinstance(value, int) and not isinstance(value, bool):
                    self.printer.print('\t{:<40}{:,}'.format(key, value))
                else:
                    self.printer.print('\t{:<40}{}'.format(key, value))

    def __print_versions(self):
        self.printer.print('Python version: {}'.format(sys.version.replace('\n', '')))
        self.printer.print('PyTorch version: {}'.format(torch.__version__))
        self.printer.print('CUDA version: {}'.format(torch.version.cuda))
