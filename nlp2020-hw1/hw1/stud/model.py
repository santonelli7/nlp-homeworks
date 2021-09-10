import torch
import torch.nn as nn
from stud.utils import log_sum_exp, sequence_mask

class CRF(nn.Module):
    '''
        Implement the CRF layer.
    '''
    def __init__(self, vocab_size):
        super(CRF, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = vocab_size + 2
        self.start_idx = vocab_size
        self.stop_idx = vocab_size + 1
        self.transitions = nn.Parameter(torch.randn(self.num_classes, self.num_classes)) # transition scores

    def forward(self, lstm_feats, lens):
        '''
            Compute the forward step for the crf.
        '''
        batch_size, _, _ = lstm_feats.shape

        # initialize the forward variables @alpha
        alpha = lstm_feats.new_full((batch_size, self.num_classes), -10000)
        alpha[:, self.start_idx] = 0

        # @lens tensor is used for build the mask to ignore the padding
        lens_copy = lens.clone()


        lstm_feats_t = lstm_feats.transpose(1, 0)
        for feat in lstm_feats_t:
            emit_scores = feat[..., None]

            # compute the score to go from a label to the others 
            scores =  emit_scores + alpha[:, None, :] + self.transitions[None, ...]

            # compute the forward variable applied log_sum_exp
            alpha_nxt = log_sum_exp(scores, 2)
            
            # consider the forward variables of the sentences which are not already terminated
            mask = (lens_copy > 0).float()
            alpha = mask[..., None] * alpha_nxt + (1 - mask[..., None]) * alpha 
            lens_copy -= 1

        # compute the forward variables even for terminals vars
        alpha = log_sum_exp(alpha + self.transitions[self.stop_idx])
        return alpha

    def viterbi_decode(self, lstm_feats, lens):
        '''
            Viterbi decode dentifies the optimal sequence of tags.
        '''
        batch_size, _, _ = lstm_feats.shape

        # initialize the viterbi variables @viterbi_vars
        viterbi_vars = lstm_feats.new_full((batch_size, self.num_classes), -10000)
        viterbi_vars[:, self.start_idx] = 0
        lens_copy = lens.clone()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lstm_feats_t = lstm_feats.transpose(1, 0)

        # viterbi_vars holds the pointers which indicate the best tags
        backpointers = torch.LongTensor().to(device)
        for feat in lstm_feats_t:

            # best previous scores and tags
            scores = viterbi_vars[:, None, :] + self.transitions[None, ...]
            best_prev_scores, best_prev_tags = scores.max(2)

            # add the emission score and add the pointer of the best previous tag in @backpointers
            vit_nxt = best_prev_scores + feat
            backpointers = torch.cat([backpointers, best_prev_tags[None, ...]], 0)

            # update viterbi variables ignoring those that are already terminated
            mask = (lens_copy > 0).float()
            viterbi_vars = mask[..., None] * vit_nxt + (1 - mask[..., None]) * viterbi_vars
            mask = (lens_copy == 1).float()
            viterbi_vars += mask[..., None] * self.transitions[self.stop_idx][None, ...]
            lens_copy -= 1

        # decode the best tag sequence
        best_scores, best_tags = viterbi_vars.max(1)
        tag_seq = torch.LongTensor(best_tags[:, None].to('cpu')).to(device)
        for backpointer in reversed(backpointers):
            rows = torch.arange(backpointer.shape[0])
            best_tags = backpointer[rows, best_tags]
            tag_seq = torch.cat([tag_seq, best_tags[:, None]], 1)

        tag_seq = tag_seq[:, :-1].flip(1)
        return tag_seq

    def transition_score(self, labels, lens):
        '''
            Compute the transition score of a sequence of tags.
        '''
        batch_size, seq_len = labels.shape

        # add '<start>' and '<stop>' indices to the labels
        starts = labels.new_full((batch_size, 1), self.start_idx)
        stops = labels.new_full((batch_size, 1), self.stop_idx)
        labels = torch.cat([starts, labels], 1)
        labels = torch.cat([labels, stops], 1)

        # get transition vector for each label
        mask = (labels>0).long()
        labels = (1 - mask) * stops + mask * labels
        transition_vec = self.transitions[labels[:, 1:]]
        
        # get the transition score for each label
        batchs, rows, _ = transition_vec.shape
        batchs = torch.arange(batchs)[:,None]
        rows = torch.arange(rows)
        transition_score = transition_vec[batchs, rows, labels[:, :-1]]

        # build the mask and consider only the score of the relevant indices (i.e. ignore '<pad>', '<start>', '<stop>')
        mask = sequence_mask(lens + 1, max_len=seq_len + 1)
        transition_score = transition_score * mask 
        score = torch.sum(transition_score, 1)
        
        return score


class LSTM_CRF(nn.Module):
    '''
        Implement the LSTM_CRF model.
    '''
    def __init__(self, params):
        super(LSTM_CRF, self).__init__()
        self.vocab_size = len(params['vocabulary'])
        self.embedding_dim = params['embedding_dim']
        self.labels_vocab = params['labels_vocabulary']
        self.hidden_dim = params['hidden_dim']
        self.bidirectional = params['bidirectional']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.embeddings = params['embeddings']
        self.clip = params['clip']
        self.crf = CRF(len(self.labels_vocab))
        self.num_classes = self.crf.num_classes

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.embeddings is not None:
            self.word_embedding.weight.data.copy_(self.embeddings)
            
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers, 
                            dropout = self.dropout if self.num_layers > 1 else 0)
        lstm_output_dim = self.hidden_dim if self.bidirectional is False else self.hidden_dim * 2
        
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(lstm_output_dim, self.num_classes)

    def _get_lstm_features(self, sentences):
        '''
            Forward step for the LSTM model.
        '''
        embeddings = self.word_embedding(sentences)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.linear(o)
        return output

    def _score_sentence(self, lstm_features, labels, lens):
        '''
            Compute the score of the features given in output by the LSTM.
        '''
        batch_size, seq_len, _ = lstm_features.shape
        batches = torch.arange(batch_size)[..., None]
        rows = torch.arange(seq_len)
        scores = lstm_features[batches, rows, labels]

        # consider only the score of the non-pad indices
        mask = sequence_mask(lens, max_len=seq_len)
        scores = scores * mask

        score = torch.sum(scores, 1)
        return score

    def score(self, sentences, labels, lens, lstm_features):
        '''
            Compute the score of a given sequence.
        '''
        transition_score = self.crf.transition_score(labels, lens)
        bilstm_score = self._score_sentence(lstm_features, labels, lens)
        score = transition_score + bilstm_score
        return score

    def forward(self, sentences, lens):
        lstm_features = self._get_lstm_features(sentences)
        preds = self.crf.viterbi_decode(lstm_features, lens)
        return preds

    def neg_log_likelihood(self, sentences, labels, lens):
        '''
            Compute the negative log likelihood used as loss function for the model
        '''
        lstm_features = self._get_lstm_features(sentences)
        forward_score = self.crf(lstm_features, lens)
        gold_score = self.score(sentences, labels, lens, lstm_features)
        return torch.mean(forward_score - gold_score)
