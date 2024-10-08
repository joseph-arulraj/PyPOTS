"""
This module contains the implementation of the DEARI model.
"""

from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _DEARI
from .data import DatasetForDEARI
from ..base import BaseNNClassifier
from ...optim.adam import Adam
from ...optim.base import Optimizer


class DEARI(BaseNNClassifier):
    """
    Parameters
    ----------
    n_steps: int
        The number of time steps in the input data.
    
    n_features: int
        The number of features in the input data.

    rnn_hidden_size: int
        The number of hidden units in the RNN.

    imputation_weight: float
        The weight of the imputation loss.

    consistency_weight: float
        The weight of the consistency loss.

    classification_weight: float
        The weight of the classification loss.

    n_classes: int
        The number of classes in the classification task.

    removal_percent: int
        The percentage of data to be removed during training.

    num_encoder_layers: int
        The number of encoder layers in the RNN.

    component_error: bool
        Whether to use component-wise error.

    is_gru: bool
        Whether to use GRU or LSTM.

    multi: int
        The number of multi-layers in the model.

    batch_size: int
        The batch size of the data used for training and validation.

    epochs: int
        The number of epochs to train the model.

    masking_mode: str
        The masking mode for the dataset. It should be one of the following: 'pygrinder' or 'deari'. If 'pygrinder', the dataset will use the masking mode from the package PyGrinder. If 'deari', the dataset will use the masking mode from the sample bidirectional masking.

    model_name: str
        The name of the model.

    dropout: float
    The dropout rate for the model to prevent overfitting. Default is 0.5.

    patience: int
        The number of epochs to wait before early stopping. If None, early stopping is not used.

    optimizer: Optimizer
        The optimizer used to train the model. Default is Adam.

    num_workers: int
        The number of subprocesses used for data loading. Setting this to `0` means that data loading is performed in the main process without using subprocesses.

    device :
        The device for the model to run on, which can be a string, a :class:`torch.device` object, or a list of devices. If not provided, the model will attempt to use available CUDA devices first, then default to CPUs.

    saving_path :
         The path for saving model checkpoints and tensorboard files during training. If not provided, models will not be saved automatically.

    model_saving_strategy :
        The strategy for saving model checkpoints. Can be one of [None, "best", "better", "all"]. "best" saves the best model after training, "better" saves any model that improves during training, and "all" saves models after each epoch. If set to None, no models will be saved.

    verbose :
        Whether to print training logs during the training process.


    """
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
                    masking_mode: str = 'pygrinder',
                    model_name: str = 'Brits_multilayer',
                    dropout: float = 0.5,
                    patience: Union[int, None] = None,
                    optimizer: Optimizer = Adam(),
                    num_workers: int = 0,
                    device: Optional[Union[str, torch.device, list]] = None,
                    saving_path: str = None,
                    model_saving_strategy: Union[str, None] = "best",
                    verbose: bool = True):
            super().__init__(
                n_classes,
                batch_size,
                epochs,
                patience,
                num_workers,
                device,
                saving_path,
                model_saving_strategy,
                verbose)
    
            self.n_steps = n_steps
            self.n_features = n_features
            self.rnn_hidden_size = rnn_hidden_size
            self.imputation_weight = imputation_weight
            self.consistency_weight = consistency_weight
            self.classification_weight = classification_weight
            self.masking_mode = masking_mode
            self.removal_percent = removal_percent
            self.num_encoderlayer = num_encoder_layers
            self.model_name = model_name
            self.component_error = component_error
            self.is_gru = is_gru
            self.multi = multi
            self.dropout = dropout
            self.device = device


            # Initialise the model
            self.model = _DEARI(
                n_steps=self.n_steps,
                n_features=self.n_features,
                rnn_hidden_size=self.rnn_hidden_size,
                imputation_weight=self.imputation_weight,
                consistency_weight=self.consistency_weight,
                classification_weight=self.classification_weight,
                n_classes=self.n_classes,
                num_encoderlayer=self.num_encoderlayer,
                component_error=self.component_error,
                is_gru=self.is_gru,
                multi=self.multi,
                model_name=self.model_name,
                dropout=self.dropout
            )
            self._send_model_to_given_device()
            self._print_model_size()

            # initialise the optimizer
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
            X_ori,
            indicating_mask,
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

        # create dataset
        self.training_set = DatasetForDEARI(
            data=train_set,
            return_y=True,
            file_type=file_type,
            removal_percent=self.removal_percent,
            masking_mode=self.masking_mode,
            training=True
        )

        self.mean_set = self.training_set.mean_set
        self.std_set = self.training_set.std_set

        # create dataloader
        train_loader = DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        # create validation dataset
        if val_set is not None:
            val_dataset = DatasetForDEARI(
                data=val_set,
                return_y=True,
                file_type=file_type,
                removal_percent=self.removal_percent,
                masking_mode=self.masking_mode,
                normalise_mean=self.mean_set,
                normalise_std=self.std_set,
            )

            # create validation dataloader
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        else:
            val_loader = None

        # set up optimizer
        self.optimizer.init_optimizer(self.model.parameters())

        # train the model
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()

        self._auto_save_model_if_necessary(confirm_saving=True)


    def predict(self, test_set: Union[dict, str], file_type: str = 'hdf5') -> dict:
        self.model.eval()
        test_set = DatasetForDEARI(
            data=test_set,
            return_y=False,
            file_type=file_type,
            removal_percent=self.removal_percent,
            masking_mode=self.masking_mode,
            normalise_mean=self.mean_set,
            normalise_std=self.std_set,
            training=False
        )

        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        classificaiton_results = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                classificaiton_results.append(results['classification_pred'])

        classification = torch.cat(classificaiton_results).cpu().detach().numpy()
        result_dict = {
            "classification": classification,
        }  
        return result_dict
    
    def classify(
            self,
            test_set,
            file_type):
        
        result_dict = self.predict(test_set, file_type)
        return result_dict['classification']
    