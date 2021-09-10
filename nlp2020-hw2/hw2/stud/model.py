import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence as pad_packed_sequence

class BiLSTM(nn.Module):
    """
        BiLSTM model to face with the SRL task.
    """
    def __init__(self, params, task_234=False):
        """
            Define the architecture of the model.
            Args:
                - task_234: boolean value to indicate if the model must do also the task predicates disambiguation
                - bert_embedding_dim: dimension of the embedding returned by Bert
                - num_pos_tags: number of distinct pos tags
                - pos_embedding_dim: dimension of the embeddings of the pos tags
                - predicates_hidden_dim: dimension of the lstm for the predicates disambiguation task
                - predicates_num_layers: number of layers for the lstm for the predicates disambiguation task
                - num_predicates: number of distinct predicates
                - num_lemmas: number of distinct lemmas
                - lemmas_embedding_dim: dimension of the embeddings of the lemmas
                - hidden_dim: dimension of the lstm for the argument classification task
                - bidirectional: boolean value to indicate if the used lstm are bidirectional
                - num_layers: number of layers for the lstm for the argument classification task
                - dropout: value of the dropout
                - num_roles: number of distinct roles
        """
        super(BiLSTM, self).__init__()
        self.task_234 = task_234
        self.bert_embedding_dim = params['bert_embedding_dim']
        self.num_pos_tags = params['num_pos_tags']
        self.pos_embedding_dim = params['pos_embedding_dim']
        self.predicates_hidden_dim = params['predicates_hidden_dim']
        self.predicates_num_layers = params['predicates_num_layers']
        self.num_predicates = params['num_predicates']
        self.num_lemmas = params['num_lemmas']
        self.lemmas_embedding_dim = params['lemmas_embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.bidirectional = params['bidirectional']
        self.num_layers = params['num_layers']
        self.dropout = params['dropout']
        self.num_roles = params['num_roles']

        self.pos_embedding = nn.Embedding(self.num_pos_tags, self.pos_embedding_dim)
        
        if self.task_234:
            # if the faced task is also predicate disambiguation, use another bilstm for this task
            self.lstm_predicates = nn.LSTM(self.bert_embedding_dim + 1 + self.pos_embedding_dim + self.lemmas_embedding_dim,
                            self.predicates_hidden_dim,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            num_layers=self.predicates_num_layers, 
                            dropout = self.dropout if self.num_layers > 1 else 0)
            predicates_embedding_dim = self.predicates_hidden_dim if self.bidirectional is False else self.predicates_hidden_dim * 2
            self.linear_predicates = nn.Linear(predicates_embedding_dim, self.num_predicates)
            roles_lstm_input_dim = self.bert_embedding_dim + 1 + self.lemmas_embedding_dim + predicates_embedding_dim
        else:
            # otherwise use the embedding to represents the predicates
            predicates_embedding_dim = self.predicates_hidden_dim
            self.predicates_embedding = nn.Embedding(self.num_predicates, predicates_embedding_dim)
            roles_lstm_input_dim = self.bert_embedding_dim + 1 + self.pos_embedding_dim + self.lemmas_embedding_dim + predicates_embedding_dim

        self.lemmas_embedding = nn.Embedding(self.num_lemmas, self.lemmas_embedding_dim)

        self.lstm = nn.LSTM(roles_lstm_input_dim,
                            self.hidden_dim,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers, 
                            dropout = self.dropout if self.num_layers > 1 else 0)
        lstm_output_dim = self.hidden_dim if self.bidirectional is False else self.hidden_dim * 2
        
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(2*lstm_output_dim, self.num_roles)

    def _duplicate_sentences(self, sentences, lens, pos_tags, lemmas, num_predicates):
        """
            Given the number of times that need to duplicate the sentences (which correspond to the number of predicates in the sentence),
            this function duplicates the sentence for that number.
            Args:
                - sentences (batch_size x sequence_length x embedding_dim)
                - lens (batch_size)
                - pos_tags (batch_size x sequence_length)
                - lemmas (batch_size x sequence_length)
                - sentence_idx (batch_size)
                - num_predicates (batch_size): the number of predicates of each sentence
            Return:
                - sentences ((batch_size + num_predicates) x sequence_length x embedding_dim): each sentence is duplicated for each predicate 
                    in the sentence
                - lens (batch_size + num_predicates): each length is duplicated for each predicate in the sentence
                - pos_tags ((batch_size + num_predicates) x sequence_length): each pos tag is duplicated for each predicate in the sentence
                - lemmas ((batch_size + num_predicates) x sequence_length): each lemma is duplicated for each predicate in the sentence
                - sentence_idx (batch_size + num_predicates): each id of the corresponding sentence is duplicated for each predicate in 
                    the sentence
        """
        sentences = torch.repeat_interleave(sentences, num_predicates, dim=0)
        lens = torch.repeat_interleave(lens, num_predicates, dim=0)
        pos_tags = torch.repeat_interleave(pos_tags, num_predicates, dim=0)
        lemmas = torch.repeat_interleave(lemmas, num_predicates, dim=0)

        return sentences, lens, pos_tags, lemmas

    def _get_number_predicates(self, predicates, predicate_null_token):
        """
            Given the predicates of every sentence return the number of predicates for each sentence and moreover both the positions of the
            predicates in a sentence and the index of that sentence in the batch. Finally, if there is no predicates in a sentence return the 
            index of that sentence with a dummy position of the predicate (usually 0).
            Args:
                - predicates (batch_size x sequence_length)
            Return:
                - num_predicates (batch_size): contains the number of predicates of each sentence
                - indices_predicates: tensor of two columns, where the second column indicates the position of the predicate in the sentence
                    corresponding to the index in the first column
                - indices_no_predicates: contains the indices of the sentences which have not predicates 
        """
        num_predicates = predicates.sum(dim=-1) if self.task_234 else (predicates != predicate_null_token).long().sum(dim=1)
        # nonzero() returns the indices of the tensor where the condition is satisfied, so the following returns the indices of the sentences
        # without predicates
        indices_no_predicates = (num_predicates == 0).nonzero().view(-1)

        # fill the indices of the sentences without predicates with 1, otherwise they will be removed in _duplicates_sentences
        num_predicates = num_predicates.scatter_(0, indices_no_predicates, 1)

        # if the condition is satisfied many times in a single sentence nonzero() returns the positions of the predicates in the sentence
        indices_predicates = (predicates == 1).nonzero() if self.task_234 else (predicates != predicate_null_token).nonzero()

        return num_predicates, (indices_predicates, indices_no_predicates)

    def _get_one_hot_predicates(self, indices_predicates, indices_no_predicates, device):
        """
            Build a tensor where in the first column there is the indices of the sentences in according to the position in the batch, 
            in the second column there is the position of the predicate in the sentence indicated by the index in the first column, 
            and in the third column there is 1 if the sentence has a predicate, 0 otherwise.
            Args:
                - indices_predicates: tensor of two column
                - indices_no_predicates: row tensor
                - device
            Return:
                - predicates_with_one_hot ((batch_size + num_predicates) x 3)
        """
        # concatenate a tensor of ones to indices_predicate to indicate that those sentences have a predicate 
        # (indices_predicates.shape[0] x 3)
        sentences_with_predicates = torch.cat([indices_predicates, torch.ones(indices_predicates.shape[0], 1, dtype=torch.long, device=device)], axis=-1)
        
        # concatenate a tensor of zeros to indices_no_predicates to indicates that those sentence does not have a predicate 
        # (indices_no_predicates.shape[0], 3)
        sentences_without_predicates = torch.cat([indices_no_predicates[...,None], torch.zeros(indices_no_predicates.shape[0], sentences_with_predicates.shape[1] - 1, dtype=torch.long, device=device)], axis=-1)

        # build a unique tensor with both the sentences with no predicates and with predicates(batch_size + num_predicates x 3)
        predicates_with_one_hot = torch.cat([sentences_with_predicates, sentences_without_predicates])

        # sort the tensor by the index of the sentences and by the position of the predicates in the sentence to restore the ordering 
        # of the sentences in the batch
        numpy_cat = predicates_with_one_hot.cpu().numpy()
        inner_sorting = np.argsort(numpy_cat[:, 1])
        numpy_cat = numpy_cat[inner_sorting]
        outer_sorting = np.argsort(numpy_cat[:, 0], kind='stable')
        numpy_cat = numpy_cat[outer_sorting]

        predicates_with_one_hot = torch.from_numpy(numpy_cat).to(device)

        return predicates_with_one_hot

    def _group_predicates(self, num_predicates, predicates):
        """
            Given the disambiguation of the splitted predicates, this function groups the predicates of each sentence in one tensor 
            picking the maximum prediction returned by the lstm for each token of the duplicated sentence.
            Args:
                - num_predicates (batch_size)
                - predicates ((batch_size + num_predicates) x sequence_length)
            Return:
                A tensor (batch_size x sequence_length) representing the classification of the predicates in the sentences.

        """
        sentences_predicates = []
        index = 0
        for i in num_predicates:
            # groups the predictions of the predicates choosing for each token the maximum prediction between the sentences resulting by 
            # the duplication
            sentences_predicates.append(torch.max(predicates[index:index+i], dim=0)[0])
            index += i
        return torch.stack(sentences_predicates)

    def _predicates_disambiguation(self, sentences, pos_tags, predicates_indicators, lemmas, lens, num_predicates):
        """
            Concatenate the pos tags to the sentences and the lemmas embeddings corresponding to the predicates and then do the 
            predicate disambiguation task.
            Args:
                - sentences ((batch_size + num_predicate) x sequence_length x (embedding_dim + 1))
                - pos_tags ((batch_size + num_predicate) x sequence_length)
                - predicates_indicators (batch_size + num_predicates) x sequence_length
                - lemmas ((batch_size + num_predicate) x sequence_length)
                - lens (batch_size + num_predicate)
                - num_predicates (batch_size)
            Return:
                The embedding of the predicates disambiguation given by the lstm and the classification of the predicates.
        """
        embeds_pos = self.pos_embedding(pos_tags)
        # ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1)) --> 
        #                               (batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + pos_embedding_dim))
        sentences = torch.cat([sentences, embeds_pos], axis=-1)

        # concatenate only the lemmas corresponding to the predicates
        lems = lemmas * predicates_indicators
        embed_lems = self.lemmas_embedding(lems)
        # (batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + pos_embedding_dim) -->
        #               (batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + pos_embedding_dim + lemmas_embedding_dim))
        sentences = torch.cat([sentences, embed_lems], axis=-1)

        sentences = pack_padded_sequence(sentences, lengths=lens, batch_first=True, enforce_sorted=False)
        # (batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + pos_embedding_dim + lemmas_embedding_dim)) -->
        #                                                        (batch_size + num_predicates) x sequence_length x predicates_embedding_dim)
        embeds_predicates, (h, c) = self.lstm_predicates(sentences)
        embeds_predicates, _ = pad_packed_sequence(embeds_predicates, batch_first=True, padding_value=0)

        # classification of the predicted predicates
        sentences = self.dropout(embeds_predicates)
        preds = self.linear_predicates(sentences)
        preds = self._group_predicates(num_predicates, preds)

        return embeds_predicates, preds

    def _expand_predicates(self, predicates, lens, indices_predicates, predicates_indicators, predicate_null_token):
        """
            Duplicate the tensor of predicates having only one predicate for each sentence.
            Args:
                - predicates (batch_size x sequence_length)
                - indices_predicates: tensor of two columns
                - predicates_indicators ((batch_size + num_predicates) x sequence_length)
                - predicates_null_token
            Return:
                - predicates ((batch_size + num_predicates) x sequence_length)
        """
        # tensor of all predicates 
        new_predicates = predicates[indices_predicates[:,0], indices_predicates[:,1]]

        # create a tensor fill with the null token
        predicates = predicates_indicators.new_full(predicates_indicators.shape, predicate_null_token, dtype=torch.long)
        
        # get the position of the of the predicate for each sentence
        indices_splitted_predicates = (predicates_indicators == 1).nonzero()

        # put the predicate in the corresponding position
        predicates[indices_splitted_predicates[:,0], indices_splitted_predicates[:,1]] = new_predicates

        # create a mask to reconstruct the padding of the sentence from the lens of the sentences
        mask_lens = (torch.arange(predicates.shape[1], dtype=lens.dtype, device=lens.device).expand(len(lens), predicates.shape[1]) < lens.unsqueeze(1)).long()
        predicates = predicates * mask_lens

        return predicates

    def _roles_disambiguation(self, sentences, lemmas, embeds_predicates, lens, predicates_with_one_hot):
        """
            Concatenate to the sentences the lemmas embeddings and the predicates embeddings and do the role disambiguation task.
            Args:
                - sentences ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1))
                - lemmas ((batch_size + num_predicates) x sequence_length x lemmas_embedding_dim): 
                - embeds_predicates ((batch_size + num_predicates) x sequence_length x predicates_embedding_dim)
                - lens (batch_size)
                - predicates_with_one_hot ((batch_size + num_predicates) x 3)
            Return:
                The classification of the roles of the sentences.
        """
        # ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1) --> 
        #   ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + 2*predicates_embedding_dim))
        sentences = torch.cat([sentences, embeds_predicates], axis=-1)
        
        # in this task all lemmas are concatenated to the sentences
        embeds_lemmas = self.lemmas_embedding(lemmas)
        # ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + 2*predicates_embedding_dim) --> 
        #                       (batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + 2*predicates_embedding_dim + lemmas_embedding_dim))
        sentences = torch.cat([sentences, embeds_lemmas], axis=-1)

        sentences = self.dropout(sentences)

        sentences = pack_padded_sequence(sentences, lengths=lens, batch_first=True, enforce_sorted=False)
        # ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + 2*predicates_embedding_dim + lemmas_embedding_dim)
        #                                                              --> (batch_size + num_predicates) x sequence_length x lstm_output_dim)
        sentences, (h, c) = self.lstm(sentences)
        sentences, _ = pad_packed_sequence(sentences, batch_first=True, padding_value=0)

        # take the hidden representation of the predicates given by the lstm
        pred_positions = predicates_with_one_hot[:,-2]
        pred_mask = predicates_with_one_hot[:,-1]
        hidden_predicates = sentences[torch.arange(sentences.shape[0]), pred_positions]
        hidden_predicates = torch.ones(sentences.shape, dtype=torch.long, device=sentences.device) * pred_mask[:,None,None] * hidden_predicates[:,None,:]

        # concatenate the hidden state of the predicate to each word of the sentence
        # ((batch_size + num_predicates) x sequence_length x lstm_output_dim 
        #                       --> (batch_size + num_predicates) x sequence_length x 2*lstm_output_dim)
        sentences = torch.cat([sentences, hidden_predicates], axis=-1)

        sentences = self.dropout(sentences)
        output = self.linear(sentences)
        return output

    def forward(self, sentences, lens, pos_tags, predicates, lemmas, predicate_null_token):
        """
            Forward step of the model.
            Args:
                - sentences (batch_size x sequence_length x embedding_dim): represents the sentences already embedded
                - lens (batch_size): contains the lengths of each sentence
                - pos_tags (batch_size x sequence_length): represents the pos_tags of each sentence
                - predicates (batch_size x sequence_length): contains the predicates of each sentence
                - lemmas (batch_size x sequence_length): represents the lemmas of each sentence
                - sentence_idx (batch_size): the ids of the sentences in the original train dataset
                - predicate_null_token: the representation of the null token in the vocabulary of predicates
        """
        # get the number of predicates for each sentence, the positions of the predicate in each sentence, and the index of the sentences 
        # with no predicates
        num_predicates, indices = self._get_number_predicates(predicates, predicate_null_token)
        indices_predicates = indices[0] 
        indices_no_predicates = indices[1] 

        # repeat the sentences with more predicates
        sentences, lens, pos_tags, lemmas = self._duplicate_sentences(sentences, lens, pos_tags, lemmas, num_predicates)
        # indicate the sentences with predicates and the sentences with no predicates (1-0)
        predicates_with_one_hot = self._get_one_hot_predicates(indices_predicates, indices_no_predicates, sentences.device)

        # create the one hot vector for the predicates starting from a tensor of zeros ((batch_size + num_predicates) x sequence_length) and
        # putting the value indicated by the third column of the predicates_with_one_hot in position indicated by the second column of
        # predicates_with_one_hot
        predicates_indicators = torch.zeros(sentences.shape[:-1], dtype=torch.long, device=sentences.device).scatter_(1, predicates_with_one_hot[:,-2][:,None], predicates_with_one_hot[:,-1][:,None])
        # ((batch_size + num_predicates) x sequence_length x embedding_dim --> 
        #                                                   (batch_size + num_predicates) x sequence_length x (embedding_dim + 1))
        out = torch.cat([sentences, predicates_indicators[..., None].float()], axis=-1)

        if self.task_234:
            # predicate disambiguation task
            embeds_predicates, preds = self._predicates_disambiguation(out, pos_tags, predicates_indicators, lemmas, lens, num_predicates)
        else:
            # duplicate also the predicates to use them for the role classification task
            predicates = self._expand_predicates(predicates, lens, indices_predicates, predicates_indicators, predicate_null_token)
            embeds_predicates = self.predicates_embedding(predicates)
            preds = None
            embeds_pos = self.pos_embedding(pos_tags)
            # ((batch_size + num_predicates) x sequence_length x (embedding_dim + 1)) --> 
            #                               (batch_size + num_predicates) x sequence_length x (embedding_dim + 1 + pos_embedding_dim))
            out = torch.cat([out, embeds_pos], axis=-1)

        # role classification task
        output = self._roles_disambiguation(out, lemmas, embeds_predicates, lens, predicates_with_one_hot)

        return output, preds    
