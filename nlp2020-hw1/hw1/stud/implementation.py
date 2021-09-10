import numpy as np
import torch
from typing import List, Tuple
from stud.utils import load_params, encode_inputs
from stud.model import LSTM_CRF

from model import Model

PATH = './model'
WEIGHTS_PATH = f'{PATH}/weights.pt'
PARAMS_PATH = f'{PATH}/params.json'

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device):
        self.device = device
        self.params = load_params(PARAMS_PATH)
        self.vocabulary = self.params['vocabulary']
        self.labels_vocab = self.params['labels_vocabulary']
        self.model = LSTM_CRF(self.params)
        self.weights = torch.load(WEIGHTS_PATH, map_location=self.device)
        self.model.load_state_dict(self.weights)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        tokens = encode_inputs(tokens, self.vocabulary).to(self.device)
        tokens_lens = torch.LongTensor([len(token[token != 0]) for token in tokens]).to(self.device)
        valid_indices = np.array((tokens != 0).tolist())
        predictions = self.model(tokens, tokens_lens)
        decoded_preds = []
        for idx, pred in enumerate(np.array(predictions.tolist())):
            decoded_pred = [self.labels_vocab[str(word)] for word in pred[valid_indices[idx]]]
            decoded_preds.append(decoded_pred)
        return decoded_preds
