import json
from typing import List
import torch
from transformers import BertTokenizer, BertModel

def read_json(path):
    """
        Read a json file.
    """
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def save_json(path, data):
    """
        Save in a json file.
    """
    with open(path, 'w') as fp:
        json.dump(data, fp)

def encode_input(sent, lemmas, pos_tags, predicates, vocab_preds, vocab_lemmas, vocab_pos_tags, bert_embedder, task_234):
    """
        Encode the input in such a way that can be fed to the model in predict function.
        Args:
            - sent: contain the tokens that compose a sentence
            - lemmas: contain the lemmas of each token
            - pos_tags: contain the pos tag of each token
            - predicates: contain the predicate represented by each token
            - vocab_preds: vocabulary of the predicates
            - vocab_lemmas: vocabulary of the lemmas
            - vocab_pos_tags: vocabulary of the pos tags
            - bert_embedder: bert embedder to embed the sentence
            - task_234: boolean value to indicate if the encode is also for the predicate disambigustion task
    """
    sentence = bert_embedder.embed_sentence([sent])[0]
    lens = torch.LongTensor([len(sent)])
    pos_tags = torch.tensor([vocab_pos_tags[pos_tag] for pos_tag in pos_tags])
    lemmas = torch.tensor([vocab_lemmas[lemma] if lemma in vocab_lemmas else vocab_lemmas['<unk>'] for lemma in lemmas])
    vocab_preds = {word: int(idx) for idx, word in vocab_preds.items()}
    predicate_null_token = vocab_preds['_']
    if task_234:
        # if the task is predicate disambiguation, predicates indicates only the position of the predicates in the sentence
        predicates = torch.LongTensor(predicates)
        positions = (predicates == 1).nonzero()
    else:
        # otherwise predicates must be encoded with the value stored in the vocabulary
        predicates = torch.tensor([vocab_preds[predicate] for predicate in predicates])
        positions = (predicates != predicate_null_token).nonzero()

    return sentence[None,...], lens, pos_tags[None,...], lemmas[None,...], predicates[None,...], positions.view(-1).tolist(), predicate_null_token

class BERTEmbedder:
    """
        Class to embed the sentences usign the pretrained model Bert.
    """
    def __init__(self, bert_model, bert_tokenizer, device):
        """
            Args:
                - bert_model: pretrained BERT model.
                - bert_tokenizer: pretrained BERT tokenizer.
                - device
        """
        super(BERTEmbedder, self).__init__()
        self.bert_model = bert_model
        self.bert_model.to(device)
        self.bert_model.eval()
        self.bert_tokenizer = bert_tokenizer
        self.device = device

    def embed_sentence(self, sentence:List[str]):
        """
            Embed the sentence in input.
            Args:
                - sentence (batch_size x sentence_length): sentence to embed
            Return:
                - embed_output (batch_size x sentence_length x embedding_dim): embedding of the sentence
        """
        # prepare the input that can be fed to bert model
        encoded_sentence, indices_subwords = self._prepare_input(sentence[0])
        with torch.no_grad():
            bert_output = self.bert_model.forward(input_ids=encoded_sentence)
      
        # take the sequence of the last four hidden states (the last element of the tuple returned by the bert model)
        # list of tensors (batch_size x num_of_splitted_words x embedding_dim)
        bert_output = list(bert_output[-1][-4:])
        bert_output.reverse()
        
        # stack the hidden states in a tensor (4 x batch_size x num_of_splitted_words x embedding_dim)
        hidden_states = torch.stack(bert_output, axis=0)
        # sum the hidden states (batch_size x num_of_splitted_words x embedding_dim)
        sum_hidden_states = torch.sum(hidden_states, axis=0)
        # merge the words splitted in subwords by the tokenizer (batch_size x sentence_length x embedding_dim)
        embed_output = self._merge_embeddings(sum_hidden_states[0], indices_subwords)
        return embed_output
  
    def _prepare_input(self, sentence:List[str]):
        """
            Prepare the input to feed to the bert model.
            Args:
                - sentence (sentence_length): sentence to encode
            Return:
                - encoded_sentence (batch_size x tokenized_sentence_length): list of ids of the words returned by bert_tokenizer
                - indices_subwords (sentence_length): list of lists where each list contain the encoded_sentence's indices that represent the subwords of a word
        """
        # tokenize the sentence and get the encoding of the sentence and the indices of the subword to merge them
        encoded_sentence, indices_subwords = self._tokenize_sentence(sentence)
        encoded_sentence = torch.LongTensor([encoded_sentence]).to(self.device)
        return encoded_sentence, indices_subwords

    def _tokenize_sentence(self, sentence:List[str]):
        """
            Tokenize the sentence.
            Args:
                - sentence (sentence_length): sentence to tokenize
            Return:
                - encoded_sentence: (tokenized_sentence_length): list of ids given by bert_tokenizer
                - indices_subwords (sentence_length): list of lists where each list contain the encoded_sentence's indices that represent the subwords of a word
        """
        # each sentence must start with the token [CLS]
        encoded_sentence = [self.bert_tokenizer.cls_token_id]
        indices_subwords = []
        for word in sentence:
            encoded_word = self.bert_tokenizer.tokenize(word)
            # save the indices of the multiple subwords which compose the word to tokenize
            indices_subwords.append(list(range(len(encoded_sentence)-1, len(encoded_sentence)+len(encoded_word)-1))) 
            encoded_sentence.extend(self.bert_tokenizer.convert_tokens_to_ids(encoded_word))
        # each sentence must end with the token [SEP]
        encoded_sentence.append(self.bert_tokenizer.sep_token_id)
        return encoded_sentence, indices_subwords

    def _merge_embeddings(self, hidden_states:List[List[float]], indices_subwords:List[List[int]]):
        """
            Merge the subwords splitted by tokenizer taking the mean of the hidden states to get the embedding representation of the sentence.
            Args:
                - hidden_states (batch_size x tokenized_sentence_length x embedding_dim): the embedding of the sentence including the subwords
                - indices_subwords (num_of_splitted_words): list of lists where each list contain the encoded_sentence's indices that represent the subwords of a word
        """
        embed_output = []
        # ignore the first and the last tokens which are respectively the [CLS] and [SEP] tokens
        hidden_states = hidden_states[1:-1 ,:]
        sentence_output = []
        for indices_to_merge in indices_subwords:
                # average the embeddings of the subwords of a word 
                sentence_output.append(torch.mean(hidden_states[indices_to_merge], axis=0))
        embed_output.append(torch.stack(sentence_output).to(self.device))
        return embed_output

class EarlyStopping:
    """
        Stop the training phase if the considerated score does not improve after a given patience.
    """
    def __init__(self, checkpoint_path, patience=3, delta=0):
        """
            Args:
                - checkpoint_path: path where the checkpoint must be saved
                - patience: how long to wait after last time score improved
                - delta: minimum change of the score
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self, score, model):
        """
            Call the early stopping.
            Args:
                - score: value on which stop the training loop
                - model: model to save its weigths
        """
        if self.best_score is None:
            # assign the best score and save the model at the end of the first epoch
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            # if the score not increase of at least delta, increment the counter and if it reach the patience early stops
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # otherwise the score is better that the saved one, so replace the best score and save the model
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """
            Save the model.
        """
        torch.save(model.state_dict(), self.checkpoint_path)
