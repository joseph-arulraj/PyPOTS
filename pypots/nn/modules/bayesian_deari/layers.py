from ...modules.deari.layers import FeatureRegression as FeatureRegression_deari, Decay as Decay_deari  



class FeatureRegression(FeatureRegression_deari):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__(input_size)

class Decay(Decay_deari):
    def __init__(self, input_size, output_size, diag=False):
        super(Decay, self).__init__(input_size, output_size, diag)





def minibatch_weight(batch_idx, num_batches):

    """Calculates the minibatch weight.
        A formula for calculating the minibatch weight is described in
        section 3.4 of the 'Weight Uncertainty in Neural Networks' paper.
        The weighting decreases as the batch index increases, this is
        because the the first few batches are influenced heavily by
        the complexity cost.
    Parameters:
        batch_idx: int -> the current batch index (from 0 to num_batches-1)
        num_batches: int -> the total number of batches
    """
    return 2 ** batch_idx / (2 ** num_batches - 1)

