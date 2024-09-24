from typing import Union
from ...imputation.deari.data import DatasetForDEARI 


class DatasetForBayesianDEARI(DatasetForDEARI):
    def __init__(self,
                    data: Union[dict, str],
                    return_X_ori: bool, 
                    return_y: bool = True,
                    file_type: str = "hdf5",
                    removal_percent: float = 0.0,
                    normalise_mean: list = [],
                    normalise_std: list = [],
                    impute_only: bool = True,
                    training: bool = True
                    ):
        
            super().__init__(
                data=data,
                return_X_ori=return_X_ori,
                return_y=return_y,
                file_type=file_type,
                removal_percent=removal_percent,
                normalise_mean=normalise_mean,
                normalise_std=normalise_std,
                impute_only=impute_only,
                training=training
            )