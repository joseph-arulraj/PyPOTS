"""
The implementation of the DEARI model.
"""

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


from .core import _DEARI
from .data import DatasetForDEARI
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...optim.adam import Adam
from ...optim.base import Optimizer

class DEARI(BaseNNImputer):
    """
    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    imputation_weight :
        weight assigned to the reconstruction loss during training

    consistency_weight :
        weight assigned to the consistency loss during training

    removal_percent :
        the percentage of data to remove from the training set

    masking_mode :
        The masking mode for the dataset. It should be one of the following: 'pygrinder' or 'deari'. If 'pygrinder', the dataset will use the masking mode from the package PyGrinder. If 'deari', the dataset will use the masking mode from the sample bidirectional masking.

    batch_size :
        the batch size for training and evaluation of the model

    epochs :
        the number of epochs to train the model

    model_name :   
        the name of the model to use for imputation

    num_encoderlayer :
        number of encoder layers for the transformer

    multi :
        The number of multi-layers in the model.

    is_gru :
        whether to use GRU or LSTM cell

    component_error :
        whether to include component estimation error in the loss

    patience :
        the number of epochs to wait before early stopping, if None, early stopping is disabled

    optimizer :
        the optimizer used for training the model. Defaults to Adam if None

    num_workers :
        The number of subprocesses used for data loading. Setting this to `0` means that data loading is performed in the main process without using subprocesses.

    device :
        The device for the model to run on, which can be a string, a :class:`torch.device` object, or a list of devices. If not provided, the model will attempt to use available CUDA devices first, then default to CPUs.

    saving_path :
         The path for saving model checkpoints and tensorboard files during training. If not provided, models will not be saved automatically.

    model_saving_strategy :
        The strategy for saving model checkpoints. Can be one of [None, "best", "better", "all"]. "best" saves the best model after training, "better" saves any model that improves during training, and "all" saves models after each epoch. If set to None, no models will be saved.

    verbose :
        Whether to print training logs during the training process.

    num_workers :
    """
    def __init__(self,
                 n_steps: int,
                 n_features: int,
                 rnn_hidden_size: int,
                 imputation_weight: float,
                 consistency_weight: float,
                 removal_percent: int,
                 batch_size: int, 
                 epochs: int, 
                 masking_mode: str = 'pygrinder',
                 model_name: str = 'Brits_multilayer',
                 num_encoder_layers: int = 2,
                 multi: int =  8,
                 is_gru: bool = False,
                 component_error: bool = False, 
                 patience: Union[int, None] = None, 
                 optimizer: Optional[Optimizer] = Adam(),
                 num_workers: int = 0, 
                 device: Union[str, torch.device, list, None] = None, 
                 saving_path: str = None, 
                 model_saving_strategy: Union[str, None] = "best", 
                 verbose: bool = True):
        super().__init__(batch_size, epochs, patience, num_workers, device, saving_path, model_saving_strategy, verbose)
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.removal_percent = removal_percent
        self.masking_mode = masking_mode
        self.num_encoder_layers = num_encoder_layers
        self.multi = multi
        self.is_gru = is_gru
        self.component_error = component_error
        self.model_name = model_name


        # Initialise the model
        self.model = _DEARI(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.num_encoder_layers,
            self.component_error,
            self.is_gru,
            self.multi,
            self.consistency_weight,
            self.imputation_weight,
            self.model_name
        )
        # Send the model to the given device
        self._send_model_to_given_device()
        self._print_model_size()

        # Initialise the optimizer
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
        ) = self._send_data_to_given_device(sample)

        # assemble input
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            }
        }

        return inputs
    
    def _assemble_input_for_validating(self, data: list) -> dict:

        # extract data
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

        # assemble input
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs
    
    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)

    def fit(self, train_set, val_set, file_type:str = "hdf5"):

        self.training_set = DatasetForDEARI(
            train_set, False, False, file_type, self.masking_mode ,self.removal_percent
            )
        
        self.mean_set = self.training_set.mean_set
        self.std_set = self.training_set.std_set

        train_loader = DataLoader(
            self.training_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
        
        val_loader = None
        
        if val_set is not None:
            self.validation_set = DatasetForDEARI(
                val_set, True, False, file_type, 
                self.masking_mode, self.removal_percent,
                self.mean_set, self.std_set,
                False
            )

            val_loader = DataLoader(
                self.validation_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )


        # set up model
        self.model = _DEARI(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.num_encoder_layers,
            self.component_error,
            self.is_gru,
            self.multi,
            self.consistency_weight,
            self.imputation_weight,
            self.model_name,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up optimizer
        self.optimizer.init_optimizer(self.model.parameters())


        # train the model
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # save the model
        self._auto_save_model_if_necessary(confirm_saving=True)


    def predict(self, test_set: Union[dict, str], file_type: str = "hdf5") -> dict:


        self.model.eval()
        test_set = DatasetForDEARI(
            test_set, True, False, file_type, 
            self.masking_mode, self.removal_percent,
            self.mean_set, self.std_set, False
        )

        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        imputation_collector = []
        x_ori_collector = []
        indicating_mask_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)
                x_ori_collector.append(inputs["X_ori"])
                indicating_mask_collector.append(inputs["indicating_mask"])


        imputed_data = torch.cat(imputation_collector).cpu().detach().numpy()
        results = {
            "imputation": imputed_data,
            "X_ori": torch.cat(x_ori_collector).cpu().detach().numpy(),
            "indicating_mask": torch.cat(indicating_mask_collector).cpu().detach().numpy(),
        }

        return results
    

    
    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]
    