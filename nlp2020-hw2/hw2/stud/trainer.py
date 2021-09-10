import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

class Trainer:
    """
        Class to train and evaluate the model.
    """
    def __init__(self, model, preds_loss_fn, roles_loss_fn, optimizer, device, early_stopping):
        """
            Args:
                - model: the model to train
                - preds_loss_fn: loss for the predicate disambiguation task
                - roles_loss_fn: loss for the role classification task
                - optimizer
                - device
                - early_stopping
        """
        self.model = model
        self.preds_loss_fn = preds_loss_fn
        self.roles_loss_fn = roles_loss_fn
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        self.early_stopping = early_stopping

    def train(self, train_dataset, valid_dataset, predicate_null_token, role_null_token, epochs=1):
        """
           Train the model.
           Args:
                - train_dataset: dataset to use for the train phase
                - valid_dataset: dataset to use for the validation phase
                - predicate_null_token: the representation of the null token in the vocabulary of predicates
                - epochs: number of epochs to train the model
            Return:
                - train_losses: the loss at each epoch computed on the train dataset
                - valid_losses: the loss at each epoch computed on the validation dataset
                - f1_scores: the f1-score at each epoch computed on the validation dataset
        """
        print('Training ...')
        train_losses = []
        valid_losses = []
        f1_scores = []
        for epoch in range(epochs):
            avg_loss, updates = 0, 0
            self.model.train()
            train_iterator = tqdm(enumerate(train_dataset), desc=f'Epoch {epoch + 1}', total=len(train_dataset))
            for step, sample in train_iterator:
                inputs = sample['sentence'].to(self.device)
                lens = sample['lens'].to(self.device)
                pos_tags = sample['pos_tags'].to(self.device)
                lemmas = sample['lemmas'].to(self.device)
                labels = [label for labels in sample['ground'] for label in labels]
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True).to(self.device)
                if self.model.task_234:
                    # if the model must do also the predicate disambiguation task, the predicates must be only the position of 
                    # them in the sentence
                    labels_predicates = sample['predicates'].to(self.device)
                    predicates = (labels_predicates != predicate_null_token).long()
                    # since labels_predicates is padded with the value of predicate_null_token, a mask built on the lengths of 
                    #the sentences is needed to pad the sentences with the token ignored by the losses
                    mask_lens = (torch.arange(labels_predicates.shape[1], dtype=lens.dtype, device=lens.device).expand(len(lens), labels_predicates.shape[1]) < lens.unsqueeze(1)).long()
                    labels_predicates = labels_predicates * mask_lens
                else:
                    predicates = sample['predicates'].to(self.device)

                self.optimizer.zero_grad()

                roles, preds = self.model(inputs, lens, pos_tags, predicates, lemmas, predicate_null_token)
                loss_roles = self.roles_loss_fn(roles.view(-1, self.model.num_roles), labels.view(-1))
                if self.model.task_234:
                    # consider also the loss for the predicate disambiaguation task 
                    loss_preds = self.preds_loss_fn(preds.view(-1, self.model.num_predicates), labels_predicates.view(-1))
                # compute a linear combination of the two loss if also the predicate disambiguation task must be done
                loss = 0.4*loss_preds + 0.6*loss_roles if self.model.task_234 else loss_roles

                # compute the loss
                avg_loss = (avg_loss * updates + loss.item()) / (updates + 1)
                updates += 1

                train_iterator.set_description(f'Epoch {epoch + 1} - Train Avg loss: {avg_loss:.4f}')

                loss.backward()
                # use clipping gradient to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                self.optimizer.step()

            # compute the validation loss and the f1-score
            val_loss, f1 = self._validation_loss(valid_dataset, epoch, predicate_null_token, role_null_token)
            
            # store the value of the metrics to make the plot of their trends
            train_losses.append(avg_loss)
            valid_losses.append(val_loss)
            f1_scores.append(f1)

            # call the early stopping to get the model with the best f1
            self.early_stopping(f1, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path))
        return train_losses, valid_losses, f1_scores

    def _validation_loss(self, valid_dataset, epoch, predicate_null_token, role_null_token):
        """
            Validate the model.
            Args:
                - valid_dataset: dataset to use for the validation 
                - epoch: current epoch
                - predicate_null_token: the representation of the null token in the vocabulary of predicates
            Return:
                - val_avg_loss: value of the loss on the validation dataset at the current epoch
                - f1: value of the f1 on the validation dataset at the current epoch
        """
        # ignore the dropout
        self.model.eval()
        val_avg_loss, val_updates = 0, 0
        all_labels = list()
        all_predictions = list()
        dev_iterator = tqdm(valid_dataset, total=len(valid_dataset))
        with torch.no_grad():
            for sample in dev_iterator:
                inputs = sample['sentence'].to(self.device)
                lens = sample['lens'].to(self.device)
                pos_tags = sample['pos_tags'].to(self.device)
                lemmas = sample['lemmas'].to(self.device)
                labels = [label for labels in sample['ground'] for label in labels]
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True).to(self.device)
                if self.model.task_234:
                    # if the model must do also the predicate disambiguation task, the predicates must be only the position of them 
                    # in the sentence
                    labels_predicates = sample['predicates'].to(self.device)
                    predicates = (labels_predicates != predicate_null_token).long()
                    # since labels_predicates is padded with the value of predicate_null_token, a mask built on the lengths of the 
                    #sentences is needed to pad the sentences with the token ignored by the losses
                    mask_lens = (torch.arange(labels_predicates.shape[1], dtype=lens.dtype, device=lens.device).expand(len(lens), labels_predicates.shape[1]) < lens.unsqueeze(1)).long()
                    labels_predicates = labels_predicates * mask_lens
                else:
                    predicates = sample['predicates'].to(self.device)

                roles, preds = self.model(inputs, lens, pos_tags, predicates, lemmas, predicate_null_token)

                loss_roles = self.roles_loss_fn(roles.view(-1, self.model.num_roles), labels.view(-1))
                if self.model.task_234:
                    # consider also the loss for the predicate disambiaguation task 
                    loss_preds = self.preds_loss_fn(preds.view(-1, self.model.num_predicates), labels_predicates.view(-1))
                # compute a linear combination of the two loss if also the predicate disambiguation task must be done
                loss = 0.4*loss_preds + 0.6*loss_roles if self.model.task_234 else loss_roles
     
                # compute the loss on the validation dataset
                val_avg_loss = (val_avg_loss * val_updates + loss.item()) / (val_updates + 1)
                val_updates += 1
                
                dev_iterator.set_description(f'Epoch {epoch + 1} - Validation Avg loss: {val_avg_loss:.4f}')

                predictions = torch.argmax(roles, dim=-1)
                predictions = predictions.view(-1)
                labels = labels.view(-1)
                # consider only the predictions that are neither pad nor null token
                valid_indices = (labels != 0) & (labels != role_null_token)
                valid_predictions = predictions[valid_indices].cpu()
                valid_labels = labels[valid_indices]
                
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())
            # compute the f1-score
            f1 = f1_score(all_labels, all_predictions, average='micro', labels=list(set(all_labels)))
        return val_avg_loss, f1

    def evaluate(self, dataset, vocab_roles, vocab_preds, predicate_null_token):
        """
            Evaluate the model using the classification report of both the tasks of predicate disambiguation and role classification
            considering the null token and without the null token.
            Args:
                - dataset: dataset to evaluate the performances of the model
                - vocab_roles: vocabulary of the roles
                - vocab_preds: vocabulary of the predicates
                - predicate_null_token: the representation of the null token in the vocabulary of predicates
        """
        # ignore the dropout
        self.model.eval()
        print('Evaluating ...')
        all_predictions = list()
        all_labels = list()
        all_predictions_preds = list()
        all_labels_preds = list()
        with torch.no_grad():
            for sample in tqdm(dataset, total=len(dataset)):
                inputs = sample['sentence'].to(self.device)
                lens = sample['lens'].to(self.device)
                pos_tags = sample['pos_tags'].to(self.device)
                lemmas = sample['lemmas'].to(self.device)
                labels = [label for labels in sample['ground'] for label in labels]
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True).to(self.device)
                if self.model.task_234:
                    # if the model must do also the predicate disambiguation task, the predicates must be only the position of them 
                    # in the sentence
                    labels_predicates = sample['predicates'].to(self.device)
                    predicates = (labels_predicates != predicate_null_token).long()
                    # since labels_predicates is padded with the value of predicate_null_token, a mask built on the lengths of the 
                    # sentences is needed to pad the sentences with the token ignored by the losses
                    mask_lens = (torch.arange(labels_predicates.shape[1], dtype=lens.dtype, device=lens.device).expand(len(lens), labels_predicates.shape[1]) < lens.unsqueeze(1)).long()
                    labels_predicates = labels_predicates * mask_lens
                else:
                    predicates = sample['predicates'].to(self.device)

                roles, preds = self.model(inputs, lens, pos_tags, predicates, lemmas, predicate_null_token)
                # take the maximum value of the representation given by the model and that is the predictions of the roles
                predictions = torch.argmax(roles, dim=-1)
                predictions = predictions.view(-1)
                labels = labels.view(-1)

                if self.model.task_234:
                    # take the maximum value of the representation given by the model and that is the predictions of the predicates
                    predictions_predicates = torch.argmax(preds, dim=-1)
                    predictions_predicates = predictions_predicates.view(-1)
                    labels_predicates = labels_predicates.view(-1)

                # ignore the padding
                valid_indices = labels != 0
                valid_predictions = predictions[valid_indices]
                valid_labels = labels[valid_indices]
                
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())

                if self.model.task_234:
                    # ignore the padding
                    valid_indices_preds = labels_predicates != 0
                    valid_predictions_pred = predictions_predicates[valid_indices_preds]
                    valid_labels_preds = labels_predicates[valid_indices_preds]
                    
                    all_predictions_preds.extend(valid_predictions_pred.tolist())
                    all_labels_preds.extend(valid_labels_preds.tolist())

        all_labels_idxs = set(all_labels)
        # take the labels predicted ignoring the null token
        labels_idxs = [idx for idx in all_labels_idxs if vocab_roles[idx] != '_']
        labels_names = [vocab_roles[idx] for idx in labels_idxs]
        # take the labels predicted
        labels_idxs_null = [idx for idx in all_labels_idxs]
        labels_names_null = [vocab_roles[idx] for idx in labels_idxs_null]

        if self.model.task_234:
            all_labels_preds_idxs = set(all_labels_preds) 
            # take the labels predicted ignoring the null token
            labels_preds_idxs = [idx for idx in all_labels_preds_idxs if vocab_preds[idx] != '_']
            labels_preds_names = [vocab_preds[idx] for idx in labels_preds_idxs]
            # take the labels predicted
            labels_preds_idxs_null = [idx for idx in all_labels_preds_idxs]
            labels_preds_names_null = [vocab_preds[idx] for idx in labels_preds_idxs_null]

        # get the classification reports for the predicate disambiguation task and role classification task both with null token 
        # and ignoring the null token
        report = classification_report(all_labels, all_predictions, labels=labels_idxs, target_names=labels_names, digits=3)
        report_null = classification_report(all_labels, all_predictions, labels=labels_idxs_null, target_names=labels_names_null, digits=3)
        preds_report = classification_report(all_labels_preds, all_predictions_preds, labels=labels_preds_idxs, target_names=labels_preds_names, digits=3) if self.model.task_234 else None
        preds_report_null = classification_report(all_labels_preds, all_predictions_preds, labels=labels_preds_idxs_null, target_names=labels_preds_names_null, digits=3) if self.model.task_234 else None
        return preds_report, preds_report_null, report, report_null
        