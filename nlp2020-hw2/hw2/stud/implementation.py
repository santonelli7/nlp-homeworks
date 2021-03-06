import json
import random
import torch
from transformers import BertTokenizer, BertModel

import numpy as np
from typing import List, Tuple

from model import Model

from stud.utils import read_json, encode_input, BERTEmbedder
from stud.model import BiLSTM

PATH = './model'
WEIGHTS_PATH = f'{PATH}/weights.pt'
WEIGHTS_234_PATH = f'{PATH}/weights_234.pt'
PARAMS_PATH = f'{PATH}/params.json'
ROLES_PATH = f'{PATH}/vocab_roles.json'
PREDS_PATH = f'{PATH}/vocab_preds.json'
LEMMAS_PATH = f'{PATH}/vocab_lemmas.json'
POS_TAGS_PATH = f'{PATH}/vocab_pos_tags.json'
TOKENIZER = f'{PATH}/bert_tokenizer/'
BERT_MODEL = f'{PATH}/bert_model/'


def build_model_34(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(device)

def build_model_234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(device, return_predicates=True)
    # raise NotImplementedError

def build_model_1234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, return_predicates=False):
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence['pos_tags']:
            prob = self.baselines['predicate_identification'][pos]['positive'] / self.baselines['predicate_identification'][pos]['total']
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)
        
        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(zip(sentence['lemmas'], predicate_identification)):
            if not is_predicate or lemma not in self.baselines['predicate_disambiguation']:
                predicate_disambiguation.append('_')
            else:
                predicate_disambiguation.append(self.baselines['predicate_disambiguation'][lemma])
                predicate_indices.append(idx)
        
        argument_identification = []
        for dependency_relation in sentence['dependency_relations']:
            prob = self.baselines['argument_identification'][dependency_relation]['positive'] / self.baselines['argument_identification'][dependency_relation]['total']
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(sentence['dependency_relations'], argument_identification):
            if not is_argument:
                argument_classification.append('_')
            else:
                argument_classification.append(self.baselines['argument_classification'][dependency_relation])
        
        if self.return_predicates:
            return {
                'predicates': predicate_disambiguation,
                'roles': {i: argument_classification for i in predicate_indices},
            }
        else:
            return {'roles': {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path='data/baselines.json'):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device, return_predicates=False):
        self.return_predicates = return_predicates
        self.device = device
        self.params = read_json(PARAMS_PATH)
        self.vocab_roles = read_json(ROLES_PATH)
        self.vocab_preds = read_json(PREDS_PATH)
        self.vocab_lemmas = read_json(LEMMAS_PATH)
        self.vocab_pos_tags = read_json(POS_TAGS_PATH)

        # get the saved bert embedder (bert base cased)
        bert_tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
        bert_model = BertModel.from_pretrained(BERT_MODEL)
        self.bert_embedder = BERTEmbedder(bert_model, bert_tokenizer, self.device)

        self.model = BiLSTM(self.params, task_234=True) if return_predicates else BiLSTM(self.params)
        weights = torch.load(WEIGHTS_234_PATH, map_location=self.device) if return_predicates else torch.load(WEIGHTS_PATH, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "pos_tags":
                            ["IN", "DT", "NN", ",", "NNP", "NNP", "CC", "DT", "NNS", "VBP", "IN", "DT", "JJ", "NNS", "IN", "VBG", "DT", "NN", "NN", "VBP", "RB", "VBN", "VBN", "."],
                        "dependency_heads":
                            ["10", "3", "1", "10", "6", "10", "6", "9", "7", "0", "10", "14", "14", "20", "14", "15", "19", "19", "16", "11", "20", "20", "22", "10"],
                        "dependency_relations":
                            ["ADV", "NMOD", "PMOD", "P", "TITLE", "SBJ", "COORD", "NMOD", "CONJ", "ROOT", "OBJ", "NMOD", "NMOD", "SBJ", "NMOD", "PMOD", "NMOD", "NMOD", "OBJ", "SUB", "TMP", "VC", "VC", "P"],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence. 
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence. 
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence. 
                    }
        """

        with torch.no_grad():
            sent = sentence['words']
            lemmas = sentence['lemmas']
            pos_tags = sentence['pos_tags']
            predicates = sentence['predicates']
            sentence, lens, pos_tags, lemmas, predicates, positions, predicate_null_token = encode_input(sent, lemmas, pos_tags, predicates, self.vocab_preds, self.vocab_lemmas, self.vocab_pos_tags, self.bert_embedder, self.return_predicates)

            roles, preds = self.model(sentence, lens, pos_tags, predicates, lemmas, predicate_null_token)

            roles = torch.argmax(roles, dim=-1)
            # decode the roles predicted
            argument_classification = [[self.vocab_roles[str(idx)] for idx in role.tolist()] for role in roles]
            roles = {positions[i]: argument_classification[i] for i in range(len(positions))}
            assert len(roles) == len(positions)

        if self.return_predicates:
            preds = torch.argmax(preds, dim=-1)
            #decode the predicates predicted
            preds = [[self.vocab_preds[str(idx)] for idx in pred.tolist()] for pred in preds][0]
            return {
                'predicates': preds,
                'roles': roles
            }
        else:
            return {
                'roles': roles
            }
