import io
import torch
import json
from tqdm.auto import tqdm
from collections import Counter

class PretrainedVec():
    '''
        Load the file of the pretrained embedding and returns a dictionary {word: embedding} 
        and the tensor (vocabulary_size, embedding_dim) of embeddings.
    '''
    def __init__(self, path, vocabulary):
        self.wtoi, self.dim = self.load_vectors(path, vocabulary)
        self.pretrained = self.build_embeddings(vocabulary)

    def load_vectors(self, path, vocabulary):
        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(fin, desc='Loading pretrained embeddings: ', total=n):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in vocabulary:
                data[word] = torch.FloatTensor([float(val) for val in tokens[1:]])
        return data, d

    def build_embeddings(self, vocabulary):
        embeddings = torch.randn(len(vocabulary), self.dim)
        initialised = 0
        for i, w in tqdm(enumerate(vocabulary), desc='Initializing embeddings: ', total=len(vocabulary)):
            if w in self.wtoi:
                initialised += 1
                embeddings[i] = self.wtoi[w]
        embeddings[vocabulary['<pad>']] = torch.zeros(self.dim, dtype=torch.float)
        print(f'initialised embeddings {initialised}')
        print(f'random initialised embeddings {len(vocabulary) - initialised}')
        return embeddings

    def __len__(self):
        return len(self.wtoi)


class Vocabulary():
    '''
        Build a vocabulary both word to index and index to word.
    '''
    def __init__(self, vocabulary, min_frequency: int=1, labels=False):
        if labels:
            self.wtoi = self.build_labels_vocabulary(vocabulary)
        else:
            self.wtoi = self.build_vocabulary(vocabulary, min_frequency)
        self.itow = {idx: word for word, idx in self.wtoi.items()}

    def build_vocabulary(self, vocabulary, min_frequency):
        vocab = dict()
        idx = 0
        for word in vocabulary:
            if vocabulary[word] > min_frequency:
                vocab[word] = idx
                idx += 1
        return vocab

    def build_labels_vocabulary(self, vocabulary):
        labels_vocab = dict()
        labels_vocab['<pad>'] = 0
        for idx, label in enumerate(vocabulary, 1):
            labels_vocab[label] = idx
        return labels_vocab

    def __len__(self):
        return len(self.wtoi)

def read_dataset(path, train, min_frequency=1):
    """
        Build a dictionary {idx: sentence} of the sentences, a dictionary {idx: label} of the 
        labels and if it is the train phase a vocabulary {word: idx} of the words in the sentences.
    """
    sentences = dict()
    ground_truth = dict()
    if train:
        vocabulary = Counter()
        vocabulary['<pad>'] = min_frequency + 1
        vocabulary['<unk>'] = min_frequency + 1
        all_labels = set()
    idx = 0
    for line in open(path):
        line = line.strip()
        if line.startswith('# '):
            sentence = []
            labels = []  
        elif line == '':
            sentences[idx] = sentence
            ground_truth[idx] = labels
            idx += 1
        else:
            _, word, label = line.split('\t')
            if train:
                vocabulary.update([word])
                if label not in all_labels:
                    all_labels.add(label)
            sentence.append(word)
            labels.append(label)
    assert len(sentences) == len(ground_truth)
    if train:
        return sentences, ground_truth, vocabulary, all_labels
    else:
        return sentences, ground_truth

def log_sum_exp(vec, dim=1):
    '''
        Compute log sum exp for the forward algorithm.
    '''
    m, _ = torch.max(vec, dim)
    return m + torch.log(torch.sum(torch.exp(vec - m[..., None]), dim))

def sequence_mask(lens, max_len):
    '''
        Build a tensor (lens, max_len) which represents the masks for the sentences. 
        The length of each sentence is in lens tensor.
    '''
    batch_size = lens.size(0)
    ranges = torch.arange(0, max_len, dtype=torch.float)
    if lens.data.is_cuda:
        ranges = ranges.cuda()
    lens_exp = lens[..., None]
    mask = (ranges < lens_exp).float()
    return mask

def save_params(data, path):
    '''
        Save the parameters of the model in a json file.
    '''
    data['embeddings'] = None
    with open(path, 'w') as file:
        json.dump(data, file)

def load_params(path):
    '''
        Load the parameters of the model from a json file.
    '''
    with open(path) as params:
        data = json.load(params)
    return data

def encode_inputs(tokens, vocabulary):
    '''
        Encode the sentences on which make inference with the ids of the words in the dictionary.
    '''
    dim = len(max(tokens, key=lambda x: len(x)))
    new_tokens = []
    for token in tokens:
        new_token = []
        for word in token:
            if word in vocabulary:
                new_token.append(vocabulary[word])
            else:
                new_token.append(vocabulary['<unk>'])
        if len(new_token) < dim:
            pad = [0 for i in range(dim - len(new_token))]
            new_token += pad
        new_tokens.append(new_token)
    return torch.LongTensor(new_tokens)