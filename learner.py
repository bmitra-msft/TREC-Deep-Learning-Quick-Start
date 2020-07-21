import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from parallel import DataParallelModel, DataParallelCriterion
from data import TRECSupervisedTrainMultiDataset, TRECSupervisedTrainDataset, TRECSupervisedTestDataset, TRECInferenceDataset
from clint.textui import progress
from factory import Factory


class Learner:

    def __init__(self, utils):
        self.utils = utils
        torch.manual_seed(self.utils.args.seed)
        self.device = torch.device(utils.args.device)
        if utils.args.device == 'cuda':
            self.num_devices = torch.cuda.device_count() 
            if utils.args.single_gpu:
                self.num_devices = min(self.num_devices, 1)
            assert self.num_devices > 0, 'no gpus available for training'
        else:
            self.num_devices = 0
        self.model = self.__get_model_instance()
        self.model_parameter_count = self.model.parameter_count()
        if self.num_devices > 1:
            self.model = DataParallelModel(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.utils.args.lr)

    def train_and_evaluate(self):
        self.utils.printer.print('starting supervised model training and evaluation of model with {:,} parameters and {} loss'.format(self.model_parameter_count, self.utils.args.loss))
        dataset = TRECSupervisedTrainDataset('train', self.utils.parent.model_utils.featurize, self.utils.parent.data_utils)
        if self.utils.args.orcas_train:
            dataset_orcas = TRECSupervisedTrainDataset('orcas', self.utils.parent.model_utils.featurize, self.utils.parent.data_utils)
            dataset = TRECSupervisedTrainMultiDataset([dataset, dataset_orcas])
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.utils.args.mb_size_train, pin_memory=(self.utils.args.device == 'cuda'))
        mb_idx = 0
        ep_idx = 0
        loss_agg = 0
        best_dev_mrr = 0
        best_val_ndcg = 0
        self.criterion = Factory.get_loss(self.utils, self.device)
        if self.num_devices > 1:
            self.criterion = DataParallelCriterion(self.criterion)
        self.model.train()
        for _, _, labels, features in self.__enumerate_infinitely(dataloader):
            self.optimizer.zero_grad()
            features = self.__move_features_to_device(features)
            labels = self.__move_features_to_device(labels)
            if isinstance(features, torch.Tensor):
                out = self.model(features)
            else:
                out = self.model(*features)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            loss_agg += loss.item()
            mb_idx += 1
            if mb_idx == self.utils.args.epoch_size:
                ep_idx += 1
                loss_agg /= self.utils.args.epoch_size
                self.utils.printer.print('epoch: {}\tloss: {:.5f}'.format(ep_idx, loss_agg), end='')
                dev_results = self.evaluate('dev', model=self.model)
                dev_mrr, _, _ = self.utils.parent.evaluate_results(dev_results, self.utils.parent.data_utils.qrels_dev)
                self.utils.printer.print('dev mrr: {:.3f}'.format(dev_mrr), end='', suppress_timestamp=True)
                val_results = self.evaluate('val', model=self.model)
                val_mrr, val_ncg, val_ndcg = self.utils.parent.evaluate_results(val_results, self.utils.parent.data_utils.qrels_val)
                self.utils.printer.print('val mrr: {:.3f}\tval ncg: {:.3f}\tval ndcg: {:.3f}'.format(val_mrr, val_ncg, val_ndcg), suppress_timestamp=True)
                if dev_mrr >= best_dev_mrr:
                    self.__save_results(val_results, 'val', 'rerank')
                    self.__save_model('dev')
                    self.best_model_dev = self.__get_model_copy(self.model)
                    best_dev_mrr = dev_mrr
                if val_ndcg >= best_val_ndcg:
                    self.__save_model('val')
                    self.best_model_val = self.__get_model_copy(self.model)
                    best_val_ndcg = val_ndcg
                mb_idx = 0
                loss_agg = 0
                self.model.train()
                if ep_idx == self.utils.args.num_epochs_train:
                    break
        val_results = self.evaluate_full_retrieval('val', model=self.best_model_dev)
        val_mrr, val_ncg, val_ndcg = self.utils.parent.evaluate_results(val_results, self.utils.parent.data_utils.qrels_val)
        self.utils.printer.print('full retrieval val mrr: {:.3f}\tfull retrieval val ncg: {:.3f}\tfull retrieval val ndcg: {:.3f}'.format(val_mrr, val_ncg, val_ndcg))
        self.__save_results(val_results, 'val', 'fullrank')
        test_results = self.evaluate('test', model=self.best_model_val)
        self.__save_results(test_results, 'test', 'rerank')
        test_results = self.evaluate_full_retrieval('test', model=self.best_model_val)
        self.__save_results(test_results, 'test', 'fullrank')

    def evaluate(self, query_set, model=None):
        dataset = TRECSupervisedTestDataset(query_set, self.utils.parent.model_utils.featurize, self.utils.parent.data_utils)
        dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=self.utils.args.mb_size_test, pin_memory=(self.utils.args.device == 'cuda'))
        if model == None:
            model = self.best_model
        if isinstance(model, DataParallelModel):
            model.module.eval()
        else:
            model.eval()
        results = {}
        with torch.no_grad():
            for _, (qids, dids, _, features) in enumerate(dataloader):
                features = self.__move_features_to_device(features)
                if isinstance(features, torch.Tensor):
                    out = model(features)
                else:
                    out = model(*features)
                if self.num_devices > 1:
                    out = torch.cat(tuple([out[i] for i in range(self.num_devices)]), dim=0)
                out = out.cpu().numpy()
                for i in range(len(qids)):
                    if qids[i] not in results:
                        results[qids[i]] = []
                    results[qids[i]].append((dids[i], out[i, 0]))
        results = {qid: sorted(docs, key=lambda x: (x[1], x[0]), reverse=True)[:self.utils.args.max_metric_pos_nodisc] for qid, docs in results.items()}
        return results

    def evaluate_full_retrieval(self, query_set, model=None):
        self.utils.printer.print('starting evaluation of model on {} under full retrieval setting'.format(query_set))
        dataset = TRECInferenceDataset(query_set, self.utils.parent.model_utils.featurize, self.utils.parent.model_utils.tokenize, self.utils.parent.data_utils)
        dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=self.utils.args.mb_size_infer, pin_memory=(self.utils.args.device == 'cuda'))
        if model == None:
            model = self.model
        if isinstance(model, DataParallelModel):
            model.module.eval()
        else:
            model.eval()
        num_query_terms = len(dataset.query_terms)
        impacts = [[] for i in range(num_query_terms)]
        with torch.no_grad():
            for dids, features in progress.bar(dataloader, expected_size=(self.utils.args.collection_size / self.utils.args.mb_size_infer) + 1):
                num_queries = len(dids)
                if num_queries < self.num_devices:
                    temp_model = self.__get_model_copy(model, num_devices_tgt=num_queries)
                    del model
                    torch.cuda.empty_cache()
                    model = temp_model
                    if isinstance(model, DataParallelModel):
                        model.module.eval()
                    else:
                        model.eval()
                features = self.__move_features_to_device(features)
                if isinstance(features, torch.Tensor):
                    out = model(features, qti_mode=True)
                else:
                    out = model(*features, qti_mode=True)
                if self.num_devices > 1:
                    out = torch.cat(tuple([out[i] for i in range(self.num_devices)]), dim=0)
                out = out.view(-1, num_query_terms).cpu().numpy()
                for i in range(len(dids)):
                    for j in range(num_query_terms):
                        score = out[i, j]
                        if score != 0:
                            impacts[j].append((dids[i], score))
        results = {}
        for qid, terms in dataset.qid_to_terms.items():
            results[qid] = {}
            for term in terms:
                term_idx = dataset.query_terms.index(term)
                for did, score in impacts[term_idx]:
                    if did not in results[qid]:
                        results[qid][did] = score
                    else:
                        results[qid][did] += score
        results = {qid: sorted([(did, score) for did, score in docs.items()], key=lambda x: (x[1], x[0]), reverse=True)[:self.utils.args.max_metric_pos_nodisc] for qid, docs in results.items()}
        return results

    def __enumerate_infinitely(self, dataloader):
        while True:
            for _, x in enumerate(dataloader):
                yield x

    def __move_features_to_device(self, features):
        if isinstance(features, torch.Tensor):
            return features.to(self.device)
        if isinstance(features, tuple) and len(features) > 0:
            return tuple(self.__move_features_to_device(feature) for feature in features)
        if isinstance(features, list) and len(features) > 0:
            return [self.__move_features_to_device(feature) for feature in features]
        if isinstance(features, dict) and len(features) > 0:
            return {k : self.__move_features_to_device(feature) for k, feature in features.items()}

    def __save_results(self, results, query_set, runtype):
        with open(os.path.join(self.utils.args.local_dir, self.utils.args.file_out_scores.format(query_set, runtype)), mode='w', encoding='utf-8') as f:
            for qid, docs in results.items():
                for i in range(len(docs)):
                    did = docs[i][0]
                    score = docs[i][1]
                    f.write('{} Q0 {} {} {} {}\n'.format(qid, did, i+1, score, self.utils.args.model))

    def __save_model(self, marker):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(self.utils.args.local_dir, self.utils.args.file_out_model.format(marker)))

    def __load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model = self.model
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __get_model_instance(self):
        return Factory.get_model(self.utils.parent.model_utils).to(self.device).float()

    def __get_model_copy(self, model_src, num_devices_tgt=0):
        model_tgt = self.__get_model_instance()
        if self.num_devices > 1:
            model_src = model_src.module
        model_tgt.load_state_dict(model_src.state_dict())
        if self.num_devices > 0:
            model_tgt = model_tgt.to(self.device)
            if self.num_devices > 1:
                if num_devices_tgt == 0:
                    model_tgt = DataParallelModel(model_tgt)
                else:
                    model_tgt = DataParallelModel(model_tgt, device_ids=list(range(num_devices_tgt)))
        return model_tgt