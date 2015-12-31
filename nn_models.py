import chainer.functions as F
from chainer.link import Chain
from fgnt.chainer_extensions.links.sequence_lstms import SequenceBLSTM
from fgnt.chainer_extensions.links.sequence_linear import SequenceLinear

from fgnt.chainer_addons import binary_cross_entropy


class MaskEstimator(Chain):
    def _propagate(self, Y, dropout=0.):
        raise NotImplemented

    def calc_masks(self, Y, dropout=0.):
        N_mask, X_mask = self._propagate(Y, dropout)
        return N_mask, X_mask

    def train_and_cv(self, Y, IBM_N, IBM_X, dropout=0.):
        N_mask_hat, X_mask_hat = self._propagate(Y, dropout)
        loss_X = F.binary_cross_entropy(X_mask_hat, IBM_X)
        loss_N = F.binary_cross_entropy(N_mask_hat, IBM_N)
        loss = (loss_X + loss_N) / 2
        return loss


class BLSTMMaskEstimator(MaskEstimator):
    def __init__(self, channels=6):
        blstm_layer = SequenceBLSTM(513, 256, normalized=True)
        relu_1 = SequenceLinear(256, 513, normalized=True)
        relu_2 = SequenceLinear(513, 513, normalized=True)
        noise_mask_estimate = SequenceLinear(513, 513, normalized=True)
        speech_mask_estimate = SequenceLinear(513, 513, normalized=True)

        super().__init__(
            blstm_layer=blstm_layer,
            relu_1=relu_1,
            relu_2=relu_2,
            noise_mask_estimate=noise_mask_estimate,
            speech_mask_estimate=speech_mask_estimate
        )

    def _propagate(self, Y, dropout=0.):
        blstm = self.blstm_layer(Y, dropout=dropout)
        relu_1 = F.clipped_relu(self.relu_1(blstm, dropout=dropout))
        relu_2 = F.clipped_relu(self.relu_2(relu_1, dropout=dropout))
        N_mask = F.sigmoid(self.noise_mask_estimate(relu_2))
        X_mask = F.sigmoid(self.speech_mask_estimate(relu_2))
        return N_mask, X_mask
