from typing import Iterable
from ...data.dataset import BaseDataset
import numpy as np
import torch
from ...data.utils import collate_fn_bidirectional, sample_loader_bidirectional, normalize_csai
from typing import Union

class DatasetForDEARI(BaseDataset):
    def __init__(self, 
                 data: Union[dict, str],
                 return_X_ori: bool, 
                 return_y: bool, 
                 file_type: str = "hdf5",
                 removal_percent: float = 0.0,
                 normalise_mean : list = [],
                 normalise_std: list = [],
                 impute_only: bool = True,
                 training: bool = True
    ):
        super().__init__(data = data, 
                         return_X_ori = return_X_ori, 
                         return_X_pred = False, 
                         return_y = return_y, 
                         file_type = file_type)
        self.removal_percent = removal_percent
        self.normalise_mean = normalise_mean
        self.normalise_std = normalise_std
        self.impute_only = impute_only
        self.training = training

        if not isinstance(self.data, str):
            self.normalized_data, self.mean_set, self.std_set = normalize_csai(self.data['X'], normalise_mean, normalise_std) 
            _data = sample_loader_bidirectional(self.normalized_data, removal_percent)
            self.processed_data = collate_fn_bidirectional(_data)
            self.forward_X = self.processed_data['values']
            self.forward_missing_mask = self.processed_data['masks']
            self.backward_missing_mask = torch.flip(self.forward_missing_mask, dims=[1])

            self.X_ori = self.processed_data['evals']
            self.indicating_mask = self.processed_data['eval_masks']
            # if self.return_y:
            #     self.y = self.processed_data['labels']

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx : int
            The index of the data to fetch.

       Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            X : tensor,
                The feature vector for model input.

            missing_mask : tensor,
                The mask indicates all missing values in X.

            delta : tensor,
                The delta matrix contains time gaps of missing values.

            label (optional) : tensor,
                The target label of the time-series sample.
        """
        sample = [
            torch.tensor([idx]),
            # forward data
            self.forward_X[idx],
            self.forward_missing_mask[idx],
            self.processed_data["deltas_f"][idx],
            # backward data
            self.backward_missing_mask[idx],
            self.processed_data["deltas_b"][idx],
        ]
        if not self.training and self.impute_only:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))  

        return {
            'sample': sample,
            'mean_set': self.mean_set,
            'std_set': self.std_set
        }
    
    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()
        X = torch.from_numpy(self.file_handle['X'][idx])
        normalized_data, mean_set, std_set = normalize_csai(X, self.normalise_mean, self.normalise_std)
        _data = sample_loader_bidirectional(normalized_data, self.removal_percent)
        processed_data = collate_fn_bidirectional(_data)

        forward_X = processed_data['values']
        forward_missing_mask = processed_data['masks']
        backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])

        X_ori = processed_data['evals']
        indicating_mask = processed_data['eval_masks']
        if self.return_y:
            y = self.y[idx].to(torch.long)

        sample = [
            torch.tensor([idx]),
            # forward data
            forward_X,
            forward_missing_mask,
            processed_data["deltas_f"],
            # backward data
            backward_missing_mask,
            processed_data["deltas_b"],
        ]
        if not self.training and self.impute_only:
            sample.extend([X_ori, indicating_mask])
        if self.return_y:
            sample.append(y)

        return {
            'sample': sample,
            'mean_set': mean_set,
            'std_set': std_set
        }