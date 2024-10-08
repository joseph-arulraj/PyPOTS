from typing import Iterable
from ...data.dataset import BaseDataset
import numpy as np
import torch
from ...data.utils import collate_fn_bidirectional, non_uniform_sample_loader_bidirectional, normalize_csai
from typing import Union
from pygrinder import mcar, fill_and_get_mask_torch
from ...data.utils import _parse_delta_torch, compute_last_obs_torch, compute_last_obs_torch_new


class DatasetForCSAI(BaseDataset):
    def __init__(self, data: Union[dict, str], 
                 return_X_ori: bool, 
                 return_y: bool, 
                 file_type: str = "hdf5",
                 removal_percent: float = 0.0,
                 increase_factor: float = 0.1,
                 compute_intervals: bool = False,
                 replacement_probabilities = None,
                 masking_code = 'pygrihnder',
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
        self.increase_factor = increase_factor
        self.compute_intervals = compute_intervals
        self.replacement_probabilities = replacement_probabilities
        self.normalise_mean = normalise_mean
        self.normalise_std = normalise_std
        self.impute_only = impute_only
        self.training = training

        if not isinstance(self.data, str):
            if masking_code == 'pygrinder':
                self.mean_set, self.std_set, self.intervals, self.replacement_probabilities = [],[],[],[]
                self.processed_data = {}
                self.normalized_data, self.mean_set, self.std_set, self.intervals = normalize_csai(self.data['X'], normalise_mean, 
                                                                                                normalise_std, compute_intervals) 
                if isinstance(self.normalized_data, np.ndarray):
                    self.normalized_data_tensor = torch.from_numpy(self.normalized_data)
                
                self.normalized_data_miss = mcar(self.normalized_data_tensor, removal_percent/100)
                self.forward_X, self.forward_missing_mask = fill_and_get_mask_torch(self.normalized_data_miss)
                
                if not self.training and self.impute_only:
                    assert (
                    "X_ori" in self.data.keys()
                    ), "The given dataset dictionary doesn't contains X_ori. Please double check."
                    
                    self.X_ori, self.X_ori_missing_mask = fill_and_get_mask_torch(self.X_ori)

                
                self.forward_X = self.forward_X.to(torch.float32)

                self.backward_X = torch.flip(self.forward_X, dims=[1])
                self.backward_missing_mask = torch.flip(self.forward_missing_mask, dims=[1])
                # compute the forward and backward delta
                self.processed_data["deltas_f"] = _parse_delta_torch(self.forward_missing_mask)
                self.processed_data["deltas_b"] = _parse_delta_torch(self.backward_missing_mask)
                # compute the forward and backward last observed value
                self.processed_data["last_obs_f"] = compute_last_obs_torch_new(self.forward_X, self.forward_missing_mask, 'forward')
                self.processed_data["last_obs_b"] = compute_last_obs_torch_new(self.forward_X, self.forward_missing_mask, 'backward')
                
                # news, self.yess = non_uniform_sample_loader_bidirectional(self.normalized_data, removal_percent,
                #                                                           replacement_probabilities, increase_factor)
                # self.nooo = collate_fn_bidirectional(news)
                # print('new and non un compare forward', torch.equal(self.processed_data["last_obs_f"], self.nooo["last_obs_f"]))
                # print('new and non un compare backward ', torch.equal(self.processed_data["last_obs_b"], self.nooo["last_obs_b"]))


            else:
                self.normalized_data, self.mean_set, self.std_set, self.intervals = normalize_csai(self.data['X'], normalise_mean, 
                                                                                                normalise_std, compute_intervals) 
                _data, self.replacement_probabilities = non_uniform_sample_loader_bidirectional(self.normalized_data, 
                                                                                                        removal_percent, 
                                                                                                        replacement_probabilities, 
                                                                                                        increase_factor)
                self.processed_data = collate_fn_bidirectional(_data)
                self.forward_X = self.processed_data['values']
                self.forward_missing_mask = self.processed_data['masks']
                self.backward_X = torch.flip(self.forward_X, dims=[1])
                self.backward_missing_mask = torch.flip(self.forward_missing_mask, dims=[1])

                self.X_ori = self.processed_data['evals']
                self.X_ori_missing_mask = self.processed_data['eval_masks']
                lf = compute_last_obs_torch_new(self.backward_X, self.backward_missing_mask, 'backward')
                print('new and non un compare backward ', torch.equal(lf, self.processed_data['last_obs_b']))
                lf2 = compute_last_obs_torch_new(self.forward_X, self.forward_missing_mask, 'forward')
                print('new and non un compare forward', torch.equal(lf2, self.processed_data['last_obs_f']))
                print('old and new compare', torch.equal(lf2, lf))

                # if self.return_y:
                #     self.y = self.processed_data['labels']
                


    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

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
            torch.tensor(idx),
            # for forward
            self.forward_X[idx],
            self.forward_missing_mask[idx],
            self.processed_data["deltas_f"][idx],
            self.processed_data["last_obs_f"][idx],
            # for backward
            self.backward_X[idx],
            self.backward_missing_mask[idx],
            self.processed_data["deltas_b"][idx],
            self.processed_data["last_obs_b"][idx],
        ]

        if not self.training and self.impute_only:
            sample.extend([self.X_ori[idx], self.X_ori_missing_mask[idx]])

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return {
            'sample': sample,
            'replacement_probabilities': self.replacement_probabilities,
            'mean_set': self.mean_set,
            'std_set': self.std_set,
            'intervals': self.intervals
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

        X = torch.from_numpy(self.file_handle["X"][idx])
        normalized_data, mean_set, std_set, intervals = normalize_csai(X, self.normalise_mean, 
                                                                           self.normalise_std, 
                                                                           self.compute_intervals)
        _data, replacement_probabilities = non_uniform_sample_loader_bidirectional(normalized_data, 
                                                                                            self.removal_percent, 
                                                                                            self.replacement_probabilities, 
                                                                                            self.increase_factor)
        processed_data = collate_fn_bidirectional(_data)
        forward_X = processed_data['values']
        forward_missing_mask = processed_data['masks']
        backward_X = torch.flip(forward_X, dims=[1])
        backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])

        X_ori = processed_data['evals']
        indicating_mask = processed_data['eval_masks']
        if self.return_y:
            y = self.y[idx].to(torch.long)


        sample = [
            torch.tensor(idx),
            # for forward
            forward_X,
            forward_missing_mask,
            processed_data["deltas_f"],
            processed_data["last_obs_f"],
            # for backward
            backward_X,
            backward_missing_mask,
            processed_data["deltas_b"],
            processed_data["last_obs_b"]
        ]

        if not self.training and self.impute_only:
            sample.extend([X_ori, indicating_mask])

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(y)

        return {
            'sample': sample,
            'replacement_probabilities': replacement_probabilities,
            'mean_set': mean_set,
            'std_set': std_set,
            'intervals': intervals
        }

