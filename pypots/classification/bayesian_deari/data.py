from typing import Union
from ...imputation.bayesian_deari.data import DatasetForBayesianDEARI as DatasetForBayesianDEARI_Imputation


class DatasetForBayesianDEARI(DatasetForBayesianDEARI_Imputation):
    def __init__(self,
                 data: Union[dict, str],
                 file_type: str = "hdf5",
                 return_y: bool = True,
                 removal_percent: float = 0.0,
                 normalise_mean: list = [],
                 normalise_std: list = [],
                 training: bool = True
                 ):
        super().__init__(
            data=data,
            return_X_ori=False,
            return_y=return_y,
            file_type=file_type,
            removal_percent=removal_percent,
            normalise_mean=normalise_mean,
            normalise_std=normalise_std,
            impute_only=False,
            training=training
       )