import torch
from torch.utils.data.dataset import Dataset
from utils import read_json, save_json
from tqdm.auto import tqdm

class Vocabulary:
    def __init__(self, data=None, path=None, special_tokens=[]):
        """
            Build a vocabulary from data or from a path.
            Args:
                - data: an iterable object cointaing data from which build the vocabulary
                - path: a path from which retrieves data to build the vocabulary
                - special_tokens: a list of special tokens (usually '<pad>' and '<unk>') to add to the top of the vocabulary
        """
        assert (data != None and path == None) or (data == None and path != None)
        if data != None:
            self.idx2word = self._build_dictionary_from_data(data, special_tokens)
            self.word2idx = {word: idx for idx, word in self.idx2word.items()}
        else:
            self.word2idx = self._build_dictionary_from_path(path, special_tokens)
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def _build_dictionary_from_data(self, data, special_tokens):
        """ 
            Build a dictionary starting from data.
        """
        dict_words = dict()
        if len(special_tokens) != 0:
            for token in special_tokens:
                dict_words[len(dict_words)] = token
        for word in data:
            if word not in dict_words.values():
                dict_words[len(dict_words)] = word
        return dict_words

    def _build_dictionary_from_path(self, path, special_tokens):
        """ 
            Build a dictionary retriving data from path.
        """
        dictionary = dict()
        if len(special_tokens) != 0:
            for token in special_tokens:
                dictionary[token] = len(dictionary)
        with open(path) as file:
            for line in file:
                line = line.strip()
                if not line.startswith('#') and not line == '':
                    _, pred = line.split('\t')
                    if pred not in dictionary:
                        dictionary[pred] = len(dictionary)
        return dictionary

    def __len__(self):
        return len(self.word2idx)


class SRLDataset(Dataset):
    """
        Class to model the dataset for the SRL task.
    """
    def __init__(self, datapath, bert_embedder, vocab_pos=None, vocab_roles=None, vocab_preds=None, vocab_lemmas=None): 
        """
            Build the dataset.
            Args:
                - datapath: path of the file where retrieve data
                - bert_embedder: bert model to encode the sentences
                - vocab_pos: vocabulary of the pos tags
                - vocab_roles: vocabulary of the roles
                - vocab_preds: vocabulary of the predicates
                - vocab_lemmas: vocabulary of the lemmas
        """
        self.bert_embedder = bert_embedder
        self.data = self._retrieve_data(datapath) #, path

        if vocab_preds is None:
            self.vocab_preds = Vocabulary(path='../../model/VA_frame_ids.tsv', special_tokens=['<pad>', '_'])
        else:
            self.vocab_preds = vocab_preds

        if vocab_roles is None:
            roles_set = set(role for item in self.data for roles in item['roles'] for role in roles)
            self.vocab_roles = Vocabulary(data=roles_set, special_tokens=['<pad>'])
        else:
            self.vocab_roles = vocab_roles

        if vocab_pos is None:
            tags_set = set(tag for item in self.data for tag in item['pos_tags'])
            self.vocab_pos = Vocabulary(data=tags_set, special_tokens=['<pad>'])
        else:
            self.vocab_pos = vocab_pos

        if vocab_lemmas is None:
            lemmas_set = set(lemma for item in self.data for lemma in item['lemmas'])
            self.vocab_lemmas = Vocabulary(data=lemmas_set, special_tokens=['<pad>', '<unk>'])
        else:
            self.vocab_lemmas = vocab_lemmas

        self._encode_data()

    def _retrieve_data(self, datapath): #, path (path represent the path where the embedding of the sentences are saved)
        """
            Retrieve data from datapath and build a dictionary with all the informations about each sentence.
        """
        file = read_json(datapath)
        data = []
        # embeds = torch.load(path) # load the saved embeddings
        for index, item in tqdm(file.items(), total=len(file), desc='Encoding data: '):
            sentence = item['words']
            embed the sentence using bert base cased
            embed = self.bert_embedder.embed_sentences([sentence])[0]
            # embed = embeds[int(index)] # take the i-th saved embedding
            # embeds.append(embed) # add the embedding to a list to save it
            sent = dict()
            if len(item['roles']) != 0:
                sent['roles'] = [roles for roles in item['roles'].values()]
            else:
                # if the sentence does not have predicates, the role of this sentence is a sequence of null token
                sent['roles'] = [['_' for word in sentence]]
            sent['predicates'] = item['predicates']
            sent['pos_tags'] = item['pos_tags']
            sent['lemmas'] = item['lemmas']
            sent['embed'] = embed
            data.append(sent)
        # torch.save(embeds, path) # save the list of the embeddings
        return data

    def _encode_data(self):
        """
            Encode data with the values in the vacabularies.
        """
        for item in self.data:
            item['roles'] = torch.tensor([[self.vocab_roles.word2idx[role] for role in roles] for roles in item['roles']])
            item['pos_tags'] = torch.tensor([self.vocab_pos.word2idx[tag] for tag in item['pos_tags']])
            item['predicates'] = torch.tensor([self.vocab_preds.word2idx[pred] for pred in item['predicates']])
            item['lemmas'] = torch.tensor([self.vocab_lemmas.word2idx[lemma] if lemma in self.vocab_lemmas.word2idx else self.vocab_lemmas.word2idx['<unk>'] for lemma in item['lemmas']])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        lemmas = self.data[idx]['lemmas']
        pos_tags = self.data[idx]['pos_tags']
        predicates = self.data[idx]['predicates']
        sentence = self.data[idx]['embed']
        ground = self.data[idx]['roles']
        return {'lemmas': lemmas,
                'pos_tags': pos_tags,
                'predicates': predicates,
                'sentence': sentence, 
                'ground': ground,
                }
