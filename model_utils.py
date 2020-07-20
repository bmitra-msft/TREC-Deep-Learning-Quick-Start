import re
import os
import csv
import sys
import math
import torch
import struct
import fasttext
import numpy as np
import krovetzstemmer
from clint.textui import progress


class NDRMUtils:

    def __init__(self, printer):
        self.printer = printer
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')
        self.stemmer = krovetzstemmer.Stemmer()
        self.stop_words = ['a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although',
                           'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'aren', 'around',
                           'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below',
                           'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes',
                           'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'couldn', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'didn', 'different',
                           'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every',
                           'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from',
                           'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'hadn', 'happens', 'hardly', 'has', 'hasn', 'have', 'haven', 'having', 'he',
                           'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored',
                           'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'isn', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know',
                           'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'll', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe',
                           'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never',
                           'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once',
                           'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please',
                           'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said',
                           'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she',
                           'should', 'shouldn', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub',
                           'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby',
                           'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took',
                           'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using',
                           'usually', 'uucp', 'v', 've', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'wasn', 'way', 'we', 'welcome', 'well', 'went', 'were', 'weren', 'what', 'whatever', 'when', 'whence', 'whenever',
                           'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within',
                           'without', 'won', 'wonder', 'would', 'would', 'wouldn', 'x', 'y', 'yes', 'yet', 'you', 'youve', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'z', 'zero']

    def parser_add_args(self, parser):
        parser.add_argument('--max_terms_query', default=20, help='maximum number of terms to consider for query (default: 20)', type=int)
        parser.add_argument('--max_terms_doc', default=4000, help='maximum number of terms to consider for long text (default: 4000)', type=int)
        parser.add_argument('--max_terms_orcas', default=2000, help='maximum number of terms to consider for long text (default: 2000)', type=int)
        parser.add_argument('--num_hidden_nodes', default=256, help='size of hidden layers (default: 256)', type=int)
        parser.add_argument('--num_encoder_layers', default=2, help='number of document encoder layers (default: 2)', type=int)
        parser.add_argument('--conv_window_size', default=31, help='window size for encoder convolution layer (default: 31)', type=int)
        parser.add_argument('--num_attn_heads', default=32, help='number of self-attention heads (default: 32)', type=int)
        parser.add_argument('--rbf_kernel_dim', default=10, help='number of RBF kernels (default: 10)', type=int)
        parser.add_argument('--rbf_kernel_pool_size', default=300, help='window size for pooling layer in RBF kernels (default: 300)', type=int)
        parser.add_argument('--rbf_kernel_pool_stride', default=100, help='stride for pooling layer in RBF kernels (default: 100)', type=int)
        parser.add_argument('--drop', default=0.2, help='dropout rate (default: 0.2)', type=float)
        parser.add_argument('--file_gen_idfs', default = 'ndrm-idfs.tsv', help = 'filename for inverse document frequencies (default: ndrm-idfs.tsv)')
        parser.add_argument('--file_gen_embeddings', default='ndrm-embeddings.bin', help='filename for fasttext embeddings (default: ndrm-embeddings.bin)')
        parser.add_argument('--orcas_field', help='use orcas data as additional document field', action='store_true')
        parser.add_argument('--no_conformer', help='use conformer model', action='store_true')

    def parser_validate_args(self, args):
        self.args = args
        assert args.max_terms_query > 0, 'maximum number of terms in query must be greater than zero'
        assert args.max_terms_doc > 0, 'maximum number of terms in document must be greater than zero'
        assert args.num_hidden_nodes % args.num_attn_heads == 0, 'number of hidden nodes should be divisible by the number of attention heads'
        assert args.drop >= 0 and args.drop < 1, 'dropout rate must be between 0 and 1'

    def setup_and_verify(self):
        self.__verify_gen_data()
        self.__preload_data_to_memory()

    def clean_text(self, s):
        s = self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', s.lower())).strip()
        s = ' '.join([self.stemmer(t) for t in s.split() if t not in self.stop_words])
        return s

    def tokenize(self, s):
        return s.split()

    def featurize(self, q, ds, infer_mode=False):
        q = self.tokenize(q)
        max_q_terms = len(q) if infer_mode else self.args.max_terms_query
        for i in range(len(ds)):
            fields = ds[i]
            other_fields = self.tokenize(' '.join(fields[:-1]))
            if self.args.orcas_field:
                orcas_field = self.tokenize(fields[-1])[:self.args.max_terms_orcas]
                ds[i] = ['<S>'] + orcas_field + other_fields + ['</S>']
            else:
                ds[i] = ['<S>'] + other_fields + ['</S>']
        feat_q, feat_mask_q = self.__get_features_lat(q, max_q_terms)
        feat_q = np.asarray(feat_q, dtype=np.int64)
        feat_mask_q = np.asarray(feat_mask_q, dtype=np.float32)
        if self.args.model != 'ndrm2':
            features = [self.__get_features_lat(doc, self.args.max_terms_doc) for doc in ds]
            feat_d = [feat[0] for feat in features]
            feat_d = np.asarray(feat_d, dtype=np.int64)
            feat_mask_d = [feat[1] for feat in features]
            feat_mask_d = np.asarray(feat_mask_d, dtype=np.float32)
        if self.args.model != 'ndrm1':
            feat_qd = [self.__get_features_exp(q, doc, max_q_terms) for doc in ds]
            feat_qd = np.asarray(feat_qd, dtype=np.float32)
            feat_idf = self.__get_features_idf(q, max_q_terms)
            feat_idf = np.asarray(feat_idf, dtype=np.float32)
            feat_dlen = self.__get_features_dlen(ds)
            feat_dlen = np.asarray(feat_dlen, dtype=np.float32)
        if self.args.model == 'ndrm1':
            return feat_q, feat_d, feat_mask_q, feat_mask_d
        if self.args.model == 'ndrm2':
            return feat_qd, feat_mask_q, feat_idf, feat_dlen
        return feat_q, feat_d, feat_qd, feat_mask_q, feat_mask_d, feat_idf, feat_dlen

    def __verify_gen_data(self):
        self.printer.print('verifying model specific input data')
        for k, file_name in vars(self.args).items():
            if k.startswith('file_gen_'):
                file_path = os.path.join(self.args.local_dir, file_name)
                if not os.path.exists(file_path):
                    if k == 'file_gen_embeddings':
                        self.__generate_embeddings(file_path)
                    elif k == 'file_gen_idfs':
                        self.__generate_idfs(file_path)

    def __generate_embeddings(self, file_path):
        self.printer.print('generating fasttext term embeddings')
        tmp_file = os.path.join(self.args.local_dir, 'tmp')
        with open(tmp_file, 'w', encoding='utf8') as f_out:
            with open(os.path.join(self.args.local_dir, self.args.file_in_qs_train), 'rt', encoding='utf8') as f_in:
                reader = csv.reader(f_in, delimiter= '\t')
                for [_, q] in reader:
                    f_out.write(q)
                    f_out.write('\n')
            with open(os.path.join(self.args.local_dir, self.args.file_in_docs), 'rt', encoding='utf8') as f_in:
                reader = csv.reader(f_in, delimiter= '\t')
                for row in reader:
                    f_out.write('\n'.join(row[1:]))
                    f_out.write('\n')
        self.printer.print('training fasttext term embeddings')
        embeddings = fasttext.train_unsupervised(tmp_file, model='skipgram', dim=self.args.num_hidden_nodes // 2, bucket=10000, minCount=100, minn=1, maxn=0, ws=10, epoch=5)
        embeddings.save_model(file_path)
        os.remove(tmp_file)

    def __generate_idfs(self, file_path):
        self.printer.print('generating inverse document frequencies for query terms')
        terms_q = set([item for sublist in [self.tokenize(q)[:self.args.max_terms_query] for q in self.parent.data_utils.qs.values()] for item in sublist])
        dfs = {term: 0 for term in terms_q}
        n = 0
        with open(os.path.join(self.args.local_dir, self.args.file_in_docs), 'rt', encoding = 'utf8') as f:
            reader = csv.reader(f, delimiter= '\t')
            for row in progress.bar(reader, expected_size=self.args.collection_size, every=(self.args.collection_size // 10000)):
                terms_d = set().union(*[field.split() for field in row[1:]])
                terms = terms_q & terms_d
                for term in terms:
                    dfs[term] += 1
                n += 1
        idfs = {k : max(math.log((n - v + 0.5) / (v + 0.5)), 0) for k,v in dfs.items()}
        idfs = {k : v for k,v in idfs.items() if v > 0}
        idfs = sorted(idfs.items(), key = lambda kv : kv[1])
        with open(file_path, 'w', encoding = 'utf8') as f:
            for (k, v) in idfs:
                f.write('{}\t{}\n'.format(k, v))

    def __preload_data_to_memory(self):
        self.printer.print('preloading model specific data to memory')
        self.vocab, self.pretrained_embeddings = self.__get_pretrained_embeddings()
        setattr(self.args, 'vocab_size', self.pretrained_embeddings.size()[0])
        self.idfs = self.__get_idfs()

    def __get_pretrained_embeddings(self):
        model = fasttext.load_model(os.path.join(self.args.local_dir, self.args.file_gen_embeddings))
        embed_size = model.get_input_matrix().shape[1] * 2
        self.__clear_line_console()
        if self.args.num_hidden_nodes != embed_size:
            self.printer.print('error: pretrained embedding size ({}) does not match specified embedding size ({})'.format(embed_size, self.args.num_hidden_nodes))
            sys.exit(0)
        pretrained_embeddings = torch.cat([torch.FloatTensor(model.get_input_matrix()), torch.FloatTensor(model.get_output_matrix())], dim=1)
        pretrained_embeddings = torch.cat([torch.zeros([3, embed_size], dtype=torch.float32), pretrained_embeddings], dim=0)
        terms = model.get_words(include_freq=False)
        vocab = {'UNK': 0, '<S>': 1, '</S>': 2}
        for i in range(len(terms)):
            vocab[terms[i]] = i + 3
        return vocab, pretrained_embeddings

    def __get_features_lat(self, terms, max_terms):
        terms = terms[:max_terms]
        num_terms = len(terms)
        num_pad = max_terms - num_terms
        features = [self.vocab.get(terms[i], self.vocab['UNK']) for i in range(num_terms)] + [0]*num_pad
        masks = [1]*num_terms + [0]*num_pad
        return features, masks

    def __get_features_exp(self, q, d, max_q_terms):
        q = q[:max_q_terms]
        features = [d.count(term) for term in q]
        pad_len = max_q_terms - len(q)
        features.extend([0]*pad_len)
        return features

    def __get_features_dlen(self, ds):
        features = [len(d) for d in ds]
        return features

    def __get_features_idf(self, terms, max_terms):
        terms = terms[:max_terms]
        num_terms = len(terms)
        num_pad = max_terms - num_terms
        features = [self.idfs.get(terms[i], 0) for i in range(num_terms)] + [0]*num_pad
        return features

    def __get_idfs(self):
        idfs = {}
        with open(os.path.join(self.args.local_dir, self.args.file_gen_idfs), 'rt', encoding = 'utf8') as f:
            reader = csv.reader(f, delimiter = '\t')
            for [term, idf] in reader:
                idfs[term] = float(idf)
        return idfs

    def __clear_line_console(self):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
