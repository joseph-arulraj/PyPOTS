from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _Bayesian_DEARI
from .data import DatasetForBayesianDEARI
from ..base import BaseNNClassifier
from ...optim.adam import Adam
from ...optim.base import Optimizer


class Bayesian_DEARI(BaseNNClassifier):
    def __init__(self,
                 n_steps: int,
                 n_features: int,
                 rnn_hidden_size: int,
                 imputation_weight: float,
                 consistency_weight: float,
                 classification_weight: float,
                 n_classes: int,
                 removal_percent: int,
                 num_encoder_layers: int,
                 component_error: bool,
                 is_gru: bool,
                 multi: int,
                 batch_size: int,
                 epochs: int,
                 unfreeze_step: int,
                 dropout: float = 0.5,
                 patience: Union[int, None] = None,
                 optimizer: Optimizer = Adam(),
                 num_workers: int = 0,
                 device: Optional[Union[str, torch.device, list]] = None,
                 saving_path: str = None,
                 model_saving_strategy: Union[str, None] = "best",
                 verbose: bool = True):
        super().__init__(n_classes, batch_size, epochs, patience, num_workers, device, saving_path, model_saving_strategy, verbose)

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.classification_weight = classification_weight
        self.removal_percent = removal_percent
        self.num_encoder_layers = num_encoder_layers
        self.multi = multi
        self.is_gru = is_gru
        self.component_error = component_error
        self.unfreeze_step = unfreeze_step
        self.dropout = dropout
        self.device = device

        # Initialise empty model
        self.model = None
        self.optimizer = optimizer

    def _assemble_input_for_training(self, data: list, training=True) -> dict:
        # extract data
        sample = data['sample']
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_missing_mask,
            back_deltas,
            labels
        ) = self._send_data_to_given_device(sample)


        # create input dictionary
        inputs = {
            'indices': indices,
            'forward': {
                'X': X,
                'missing_mask': missing_mask,
                'deltas': deltas,
            },
            'backward': {
                'X': X,
                'missing_mask': back_missing_mask,
                'deltas': back_deltas,
            },
            'labels': labels,
        }

        return inputs
    
    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)
    
    def _assemble_input_for_testing(self, data: list) -> dict:
        
        sample = data['sample']
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_missing_mask,
            back_deltas,
        ) = self._send_data_to_given_device(sample)

        inputs = {
            'indices': indices,
            'forward': {
                'X': X,
                'missing_mask': missing_mask,
                'deltas': deltas,
            },
            'backward': {
                'X': X,
                'missing_mask': back_missing_mask,
                'deltas': back_deltas,
            },
        }

        return inputs
    
    def fit(self, train_set, val_set=None, file_type:str = 'hdf5'):

        self.training_set = DatasetForBayesianDEARI(
            data=train_set,
            file_type=file_type,
            return_y=True,
            removal_percent=self.removal_percent,
            training=True
        )

        train_loader = DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        if val_set is not None:
            self.val_set = DatasetForBayesianDEARI(
                data=val_set,
                file_type=file_type,
                return_y=True,
                removal_percent=self.removal_percent,
                normalise_mean=self.training_set.mean_set,
                normalise_std=self.training_set.std_set,
                training=False
            )

            val_loader = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )   

        else:
            val_loader = None

        num_batches = len(train_loader) * self.epochs
        self.model = _Bayesian_DEARI(
            n_steps=self.n_steps,
            n_features=self.n_features,
            rnn_hidden_size=self.rnn_hidden_size,
            imputation_weight=self.imputation_weight,
            consistency_weight=self.consistency_weight,
            classification_weight=self.classification_weight,
            n_classes=self.n_classes,
            num_encoderlayer=self.num_encoder_layers,
            component_error=self.component_error,
            is_gru=self.is_gru,
            multi=self.multi,
            dropout=self.dropout,
            device=self.device,
            number_of_batches=num_batches,
            unfreeze_step=self.unfreeze_step
        )

        self._send_model_to_given_device()
        self._print_model_size()

        # set up optimizer
        self.optimizer.init_optimizer(self.model.parameters())

        # train the model
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()

        # save the model
        self._auto_save_model_if_necessary(confirm_saving=True)


    def predict(self, test_set: Union[dict, str], file_type: str = "hdf5") -> dict:
        
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please train the model first.")
        
        self.model.eval()
        test_set = DatasetForBayesianDEARI(
            data=test_set,
            return_y=False,
            file_type=file_type,
            removal_percent=self.removal_percent,
            normalise_mean=self.training_set.mean_set,
            normalise_std=self.training_set.std_set,
            training=False
        )

        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        classification_results = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                classification_results.append(results['classification_pred'])

        predictions = torch.cat(classification_results).cpu().detach().numpy()

        result_dict = {
            "classification": predictions,
        }

        return result_dict
    
    def classify(
            self,
            test_set,
            file_type):
        
        result_dict = self.predict(test_set, file_type)
        return result_dict['classification']
