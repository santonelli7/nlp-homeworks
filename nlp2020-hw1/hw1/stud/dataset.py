import torch
from torch.utils.data import Dataset
from utils import read_dataset, Vocabulary

class Dataset(Dataset):
    def __init__(self, input_file, train=False, vocab=None, labels_vocab=None, min_frequency=1):
        self.vocab = vocab
        self.labels_vocab = labels_vocab
        self.get_data_from_file(input_file, train, min_frequency)
        self.encode_data()

    def get_data_from_file(self, input_file, train, min_frequency):
        """
            Get the data from the file @input_file and build the vocabulary if the dataset to be built is for training task
        """
        if train:
            self.sentences, self.ground_truth, vocabulary, all_labels = read_dataset(input_file, train, min_frequency)
            self.vocab = Vocabulary(vocabulary, min_frequency)
            self.labels_vocab = Vocabulary(all_labels, labels=True)
        else:
            self.sentences, self.ground_truth = read_dataset(input_file, train)

    def longest_sentence(self):
        """
            Return the length of the longest sentence in the train set
        """
        return len(max(self.sentences.values(), key=lambda x: len(x)))

    def encode_data(self):
        """
            Fill the sentences with the tag '<pad>', i.e. 0, to have homogeneous vectors
        """
        dim = self.longest_sentence()
        for index, sentence in self.sentences.items():
            enc_sent = self.encode_sentence(sentence)
            enc_ground = self.encode_ground(self.ground_truth[index])
            if len(sentence) < dim:
                padding_sent = torch.zeros(dim, dtype=torch.long)
                padding_gr = torch.zeros(dim, dtype=torch.long)
                padding_sent[:len(enc_sent)] = enc_sent
                enc_sent = padding_sent
                padding_gr[:len(enc_ground)] = enc_ground
                enc_ground = padding_gr 
            self.sentences[index] = enc_sent
            self.ground_truth[index] = enc_ground
            assert len(self.sentences[index]) == len(self.ground_truth[index])

    def encode_sentence(self, raw_sentence):
        """
            Encode the sentence with the representation in the vocabulary
        """
        sentence = []
        for word in raw_sentence:
            if word in self.vocab.wtoi:
                sentence.append(self.vocab.wtoi[word])
            else:
                sentence.append(self.vocab.wtoi['<unk>'])
        return torch.LongTensor(sentence)

    def encode_ground(self, raw_ground):
        """
            Encode the ground truth with the representation in the vocabulary
        """
        return torch.LongTensor([self.labels_vocab.wtoi[label] for label in raw_ground])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        ground = self.ground_truth[idx]
        return {"sentence": sentence, "ground": ground}
