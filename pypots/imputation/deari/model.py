
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
    def __init__(self,
                 n_steps: int,
                 n_features: int,
                 rnn_hidden_size: int,
                 imputation_weight: float,
                 consistency_weight: float,
                 removal_percent: int,
                 batch_size: int, 
                 epochs: int, 
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
        self.num_encoder_layers = num_encoder_layers
        self.multi = multi
        self.is_gru = is_gru
        self.component_error = component_error
        self.model_name = model_name


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
            train_set, False, False, file_type, self.removal_percent
            )
        
        treain_loader = DataLoader(
            self.training_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
        
        if val_set is not None:
            self.validation_set = DatasetForDEARI(
                val_set, False, False, file_type, self.removal_percent,
                self.training_set.mean_set, self.training_set.std_set,
                True, False
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
                self.device
            )
            self._send_model_to_given_device()
            self._print_model_size()

            # set up optimizer
            self.optimizer.init_optimizer(self.model.parameters())


            # train the model
            self._train_model(treain_loader, val_loader)
            self.model.load_state_dict(self.best_model_dict)
            self.model.eval()  # set the model as eval status to freeze it.

            # save the model
            self._auto_save_model_if_necessary(confirm_saving=True)


    def predict(self, test_set: Union[dict, str], file_type: str = "hdf5") -> dict:

        if self.model is None:
            raise ValueError("The model has not been trained yet. Please train the model first.")
        
        self.model.eval()
        test_set = DatasetForDEARI(
            test_set, False, False, file_type, self.removal_percent,
            self.training_set.mean_set, self.training_set.std_set, True, False
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
                imputation_collector.append(results["imputed_data"])
                x_ori_collector.append(inputs["X_ori"])
                indicating_mask_collector.append(inputs["indicating_mask"])


        imputed_data = torch.cat(imputation_collector, dim=0)
        results = {
            "imputation": imputed_data,
            "X_ori": torch.cat(x_ori_collector, dim=0),
            "indicating_mask": torch.cat(indicating_mask_collector, dim=0)
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