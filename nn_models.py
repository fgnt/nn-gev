from chainer import Variable
from chainer.link import Chain
from chainer.links import BatchNormalization
from chainer.links import LSTM
from chainer.links import Linear

from fgnt.chainer_addons import binary_cross_entropy


class MaskEstimator(Chain):
    def _propagate(self, Y, train=False, dropout=0.):
        raise NotImplemented

    def calc_masks(self, Y, train=False, dropout=0.):
        Y_var = Variable(Y, not train)
        if not self._cpu:
            Y_var.to_gpu()
        N_mask, X_mask = self._propagate(Y, train, dropout)
        return N_mask, X_mask

    def train_and_cv(self, Y, IBM_N, IBM_X, train=True, dropout=.5):
        mask_N, mask_X = self.calc_masks(Y, train=train, dropout=dropout)
        loss = binary_cross_entropy(mask_N, IBM_N)
        loss += binary_cross_entropy(mask_X, IBM_X)
        return loss


class BLSTMMaskEstimator(MaskEstimator):
    def __init__(self):
        super().__init__(
            batch_norm=BatchNormalization(513),
            lstm_fw=LSTM(513, 256),
            lstm_bw=LSTM(513, 256),
            ff_1=Linear(256, 1024),
            ff_2=Linear(1024, 1024),
            ff_3=Linear(1024, 1024),
            ff_X_mask=Linear(1024, 513),
            ff_N_mask=Linear(1024, 513)
        )

    def _propagate(self, Y, train=False, dropout=0.):
        Y_norm = self.batch_norm(Y, test=not train, finetune=train)
        lstm_in = F.dropout(Y_norm, dropout, train=train)
        self.lstm_fw.reset_state()
        self.lstm_bw.reset_state()
        T = lstm_in.data.shape[0]
        B = lstm_in.data.shape[1]
        F = lstm_in.data.shape[2]
        lstm_fw_out = list()
        for frame in F.split_axis(lstm_in, T, axis=0):
            lstm_fw_out.append(self.lstm_fw(frame))
        lstm_bw_out = list()
        for frame in reversed(F.split_axis(lstm_in, T, axis=0)):
            lstm_bw_out.append(self.lstm_bw(frame))
        lstm_fw_out = F.concat(lstm_fw_out, axis=0)
        lstm_bw_out = F.concat(reversed(lstm_bw_out), axis=0)
        blstm_out = lstm_fw_out + lstm_bw_out
        blstm_out = F.reshape(blstm_out, (T * B, F))
        ff_1 = F.clipped_relu(self.ff_1(F.dropout(blstm_out, dropout, train)))
        ff_2 = F.clipped_relu(self.ff_2(F.dropout(ff_1, dropout, train)))
        ff_3 = F.clipped_relu(self.ff_3(F.dropout(ff_2, dropout, train)))
        mask_X = F.reshape(F.sigmoid(self.ff_X_mask(ff_3)), (T, B, 513))
        mask_N = F.reshape(F.sigmoid(self.ff_X_mask(ff_3)), (T, B, 513))
        return mask_N, mask_X
