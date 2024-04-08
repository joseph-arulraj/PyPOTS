"""
The package of the partially-observed time-series imputation model M-RNN.

Refer to the paper
`Jinsung Yoon, William R. Zame, and Mihaela van der Schaar.
Estimating missing data in temporal data streams using multi-directional recurrent neural networks.
IEEE Transactions on Biomedical Engineering, 66(5):14771490, 2019.
<https://arxiv.org/pdf/1711.08742>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import MRNN


__all__ = [
    "MRNN",
]
