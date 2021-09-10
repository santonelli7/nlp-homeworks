import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

class Trainer():
    """
        Utility class to train and evaluate a model
    """
    def __init__(self, model: nn.Module, optimizer, device):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer

    def train(self, train_dataset:Dataset, valid_dataset:Dataset, epochs=1):
        """
            Train the model and compute the train loss and the validation loss at each epoch
        """
        assert epochs > 1 and isinstance(epochs, int)
        print('Training ...')
        train_loss = 0.0
        for epoch in range(epochs):

            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc=f'Epoch {epoch + 1}', total=len(train_dataset)):
                inputs = sample['sentence'].to(self.device)
                lens = torch.LongTensor([len(sentence[sentence != 0]) for sentence in inputs]).to(self.device)
                labels = sample['ground'].to(self.device)
                self.model.zero_grad()

                loss = self.model.neg_log_likelihood(inputs, labels, lens)
                loss.backward()

                # set the clip step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.clip)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch + 1, avg_epoch_loss))
            valid_loss = self.validation_loss(valid_dataset)
            print('\t[E: {:2d}] valid loss = {:0.4f}'.format(epoch + 1, valid_loss))
        
        valid_loss = self.validation_loss(valid_dataset)
        print(f'Validation loss = {valid_loss}')
        print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    def validation_loss(self, valid_dataset):
        """
            Compute the loss on the validation dataset
        """
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['sentence'].to(self.device)
                lens = torch.LongTensor([len(sentence[sentence != 0]) for sentence in inputs]).to(self.device)
                labels = sample['ground'].to(self.device)

                loss = self.model.neg_log_likelihood(inputs, labels, lens)
                valid_loss += loss.item()
        
        return valid_loss / len(valid_dataset)
    

    def evaluate(self, valid_dataset, vocab_labels):
        """
            Returns the metrics to evaluate the model, in particular the classification report and the 
            confusion matrix
        """
        self.model.eval()
        print('Evaluating ...')
        all_predictions = list()
        all_labels = list()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, total=len(valid_dataset)):
                inputs = sample['sentence'].to(self.device)
                lens = torch.LongTensor([len(sentence[sentence != 0]) for sentence in inputs]).to(self.device)
                labels = sample['ground'].to(self.device)
                predictions = self.model(inputs, lens)
                predictions = predictions.view(-1)
                labels = labels.view(-1)
                valid_indices = labels != 0
                
                valid_predictions = predictions[valid_indices]
                valid_labels = labels[valid_indices]
                
                all_predictions.extend(valid_predictions.tolist())
                all_labels.extend(valid_labels.tolist())

        labels_names = [label for label in vocab_labels.keys()][1:]
        report = classification_report(all_labels, all_predictions, target_names=labels_names)
        cm = confusion_matrix(all_labels, all_predictions)
        return report, cm
