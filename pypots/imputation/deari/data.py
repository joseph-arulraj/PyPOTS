import copy
from typing import Iterable, Union
from ...data.dataset import BaseDataset
import numpy as np
import torch
from ..csai.data import normalize_csai, non_uniform_sample, parse_delta, compute_last_obs   
from pygrinder import mcar, fill_and_get_mask_torch
from ...data.utils import _parse_delta_torch

# def compute_last_obs_tensor(data, masks):
#     """
#     Compute the last observed values for each time step for the whole tensor.
    
#     Parameters:
#     - data (torch.Tensor): Input tensor of shape [B, T, D].
#     - masks (torch.Tensor): Binary masks indicating where data is not NaN, of shape [B, T, D].

#     Returns:
#     - last_obs (torch.Tensor): Last observed values tensor of shape [B, T, D].
#     """ 

#     def compute_last_obs_single_batch(data, masks):
#         """
#         Compute the last observed values for each time step for a single batch.
#         """
#         T, D = masks.shape
#         last_obs = torch.full((T, D), np.nan)  # Initialize last observed values with NaNs for a single batch
#         last_obs_val = torch.full((D,), np.nan)  # Initialize last observed values for first time step with NaNs

#         for t in range(1, T):  # Start from t=1, keeping first row as NaN
#             mask = masks[t - 1].bool()
#             # Update last observed values based on previous time step
#             last_obs_val[mask] = data[t - 1, mask]
#             # Assign last observed values to the current time step
#             last_obs[t] = last_obs_val 
        
#         return last_obs


    # B, T, D = masks.shape
    # last_obs_batch = torch.full((B, T, D), np.nan)  # Initialize last observed values with NaNs for a whole batch
    # # loop over each batch
    # for b in range(B):
    #     last_obs_batch[b] = compute_last_obs_single_batch(data[b], masks[b])

    # return last_obs_batch


def sample_loader_bidirectional( data, removal_percent):
    """
    Function to process the data by randomly eliminating a certain percentatge of observed values as the imputation ground-truth, 
    and compute the necessary features such as masks, deltas, last_obs, evals, eval_masks, etc.

    Parameters:
    ----------
    data (np.ndarray): 
        Input data of shape [N, T, D].

    removal_percent (float): 
        Percentage of observed values to be randomly eliminated as the imputation ground-truth.

    Returns:
    -------
    tensor_dict (dict): 
        Dictionary containing the processed data in the form of PyTorch tensors. The dictionary contains the following
        keys: 'values', 'last_obs_f', 'last_obs_b', 'masks', 'evals', 'eval_masks', 'deltas_f', 'deltas_b'.

    """
    # Random seed
    np.random.seed(1)
    torch.manual_seed(1)

    # Get Dimensionality
    [N, T, D] = data.shape

    # Reshape
    data = data.reshape(N, T*D)

    recs = []
    number = 0
    masks_sum = np.zeros(D)
    eval_masks_sum = np.zeros(D)
    for i in range(N):
        try:
            values = copy.deepcopy(data[i])
            if removal_percent != 0:
                # randomly eliminate 10% values as the imputation ground-truth
                indices = np.where(~np.isnan(data[i]))[0].tolist()
                indices = np.random.choice(indices, len(indices) * removal_percent // 100)
                values[indices] = np.nan

            # Compute masks and eval_masks
            masks = ~np.isnan(values)
            eval_masks = (~np.isnan(values)) ^ (~np.isnan(data[i]))
            evals = data[i].reshape(T, D)
            values = values.reshape(T, D)
            masks = masks.reshape(T, D)
            eval_masks = eval_masks.reshape(T, D)
            rec = {}
            # Compute forward and backward deltas
            deltas_f = parse_delta(masks)
            deltas_b = parse_delta(masks[::-1, :])

            # Compute last observed values for forward and backward directions
            last_obs_f = compute_last_obs(values, masks, direction='forward')
            last_obs_b = compute_last_obs(values, masks, direction='backward')

            # Append the processed data to the list
            recs.append({
                'values': np.nan_to_num(values),
                'last_obs_f': np.nan_to_num(last_obs_f),
                'last_obs_b': np.nan_to_num(last_obs_b),
                'masks': masks.astype('int32'),
                'evals': np.nan_to_num(evals),
                'eval_masks': eval_masks.astype('int32'),
                'deltas_f': deltas_f,
                'deltas_b': deltas_b
            })
            number += 1
            masks_sum += np.sum(masks, axis=0)
            eval_masks_sum += np.sum(eval_masks, axis=0)

        except Exception as e:
            print(f"An exception occurred: {type(e).__name__}")
            continue

        
        # Convert the records to pytorch tensors
        tensor_dict = {
            'values': torch.FloatTensor(np.array([rec['values'] for rec in recs])),
            'last_obs_f': torch.FloatTensor(np.array([rec['last_obs_f'] for rec in recs])),
            'last_obs_b': torch.FloatTensor(np.array([rec['last_obs_b'] for rec in recs])),
            'masks': torch.FloatTensor(np.array([rec['masks'] for rec in recs])),
            'evals': torch.FloatTensor(np.array([rec['evals'] for rec in recs])),
            'eval_masks': torch.FloatTensor(np.array([rec['eval_masks'] for rec in recs])),
            'deltas_f': torch.FloatTensor(np.array([rec['deltas_f'] for rec in recs])),
            'deltas_b': torch.FloatTensor(np.array([rec['deltas_b'] for rec in recs]))
        }

    return tensor_dict



class DatasetForDEARI(BaseDataset):
    """Dataset class for DEARI model.
    
    Parameters:
    ----------
    data (Union[dict, str]):
        The dataset for model input, which can be either a dictionary or a path string to a data file. If it's a dictionary, `X` should be an array-like structure with shape [n_samples, sequence length (n_steps), n_features], containing the time-series data, and it can have missing values. Optionally, the dictionary can include `y`, an array-like structure with shape [n_samples], representing the labels of `X`. If `data` is a path string, it should point to a data file (e.g., h5 file) that contains key-value pairs like a dictionary, including keys for `X` and possibly `y`.
    
    return_X_ori (bool):
        Whether to return the original input data `X`in the dataset. If True, the dataset will return the original input data `X` as one of the outputs. If False, the dataset will not return the original input data `X`.
    
    return_y (bool):
        Whether to return the target labels `y` in the dataset. If True, the dataset will return the target labels `y` as one of the outputs. If False, the dataset will not return the target labels `y`.

    file_type (str):
        The type of the data file, which should be one of the following: 'hdf5'.

    masking_mode (str):
        The masking mode for the dataset. It should be one of the following: 'pygrinder' or 'deari'. If 'pygrinder', the dataset will use the masking mode from the package PyGrinder. If 'deari', the dataset will use the masking mode from the sample bidirectional masking.
    
    removal_percent (float):
        Percentage of observed values to be randomly eliminated as the imputation ground-truth.

    normalise_mean (list):
        List of mean values for normalizing the input data. If not provided, the mean values will be computed from the input data.

    normalise_std (list):
        List of standard deviation values for normalizing the input data. If not provided, the standard deviation values will be computed from the input data.

    training (bool):
        Whether the dataset is used for training. If True, the dataset will return the target labels `y` as one of the outputs. If False, the dataset will not return the target labels `y`.    
    """
    def __init__(self, 
                 data: Union[dict, str],
                 return_X_ori: bool, 
                 return_y: bool, 
                 file_type: str = "hdf5",
                 masking_mode: str = "pygrinder",
                 removal_percent: float = 0.0,
                 normalise_mean : list = [],
                 normalise_std: list = [],
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
        self.training = training

        if not isinstance(self.data, str):
            self.normalized_data, self.mean_set, self.std_set = normalize_csai(self.data['X'], 
                                                                               normalise_mean, 
                                                                               normalise_std
                                                                               ) 
            if masking_mode == 'pygrinder':
                self.processed_data = {}
                if isinstance(self.normalized_data, np.ndarray):
                    self.normalized_data_tensor = torch.from_numpy(self.normalized_data)
                
                # Introduce artificial missing values
                self.normalized_data_miss = mcar(self.normalized_data_tensor, removal_percent/100)
                # Fill missing values and get missing mask
                self.forward_X, self.forward_missing_mask = fill_and_get_mask_torch(self.normalized_data_miss)#

                # backward missing mask
                self.backward_missing_mask = torch.flip(self.forward_missing_mask, dims=[1])

                self.forward_X = self.forward_X.to(torch.float32)
                # Compute forward and backward deltas
                self.processed_data["deltas_f"] = _parse_delta_torch(self.forward_missing_mask)
                self.processed_data["deltas_b"] = _parse_delta_torch(self.backward_missing_mask)

            else:
                self.processed_data = sample_loader_bidirectional(self.normalized_data, 
                                                                removal_percent
                                                                )
                self.forward_X = self.processed_data['values']
                self.forward_missing_mask = self.processed_data['masks']
                self.backward_missing_mask = torch.flip(self.forward_missing_mask, dims=[1])

                self.X_ori = self.processed_data['evals']
                self.indicating_mask = self.processed_data['eval_masks']

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
        if not self.training:
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
        processed_data = sample_loader_bidirectional(normalized_data, self.removal_percent)

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
        if self.return_X_ori:
            sample.extend([X_ori, indicating_mask])
        if self.return_y:
            sample.append(y)

        return {
            'sample': sample,
            'mean_set': mean_set,
            'std_set': std_set
        }
    