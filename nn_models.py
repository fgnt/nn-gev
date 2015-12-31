import chainer.functions as chainer_func
from chainer.link import Chain
from chainer.links import BatchNormalization
from chainer.links import LSTM
from chainer.links import Linear

from fgnt.chainer_addons import binary_cross_entropy


class MaskEstimator(Chain):
    def _propagate(self, Y, train=False, dropout=0.):
        raise NotImplemented

    def calc_masks(self, Y, train=False, dropout=0.):
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
        self._3d_shape = (0, 0, 0)

    def _to_2d(self, var):
        org_shape = var.data.shape
        return chainer_func.reshape(
            var, (org_shape[0] * org_shape[1], org_shape[2])), org_shape

    def _to_3d(self, var, org_shape):
        return chainer_func.reshape(
            var, (org_shape[0], org_shape[1], var.data.shape[1])
        )

    def _propagate(self, Y, train=False, dropout=0.):
        Y_2d, y_shape = self._to_2d(Y)
        Y_norm_2d = self.batch_norm(Y_2d, test=not train, finetune=train)
        Y_norm = self._to_3d(Y_norm_2d, y_shape)
        lstm_in = chainer_func.dropout(Y_norm, dropout, train=train)
        self.lstm_fw.reset_state()
        self.lstm_bw.reset_state()
        lstm_fw_out = list()
        T = lstm_in.data.shape[0]
        for frame in chainer_func.split_axis(lstm_in, T, axis=0):
            frame = chainer_func.reshape(
                frame, (frame.data.shape[1], frame.data.shape[2]))
            lstm_out = self.lstm_fw(frame)
            lstm_out = chainer_func.reshape(
                lstm_out, (1, lstm_out.data.shape[0], lstm_out.data.shape[1])
            )
            lstm_fw_out.append(lstm_out)
        lstm_bw_out = list()
        for frame in reversed(chainer_func.split_axis(lstm_in, T, axis=0)):
            frame = chainer_func.reshape(
                frame, (frame.data.shape[1], frame.data.shape[2]))
            lstm_out = self.lstm_fw(frame)
            lstm_out = chainer_func.reshape(
                lstm_out, (1, lstm_out.data.shape[0], lstm_out.data.shape[1])
            )
            lstm_bw_out.append(lstm_out)
        lstm_fw_out = chainer_func.concat(lstm_fw_out, axis=0)
        lstm_bw_out = chainer_func.concat(reversed(lstm_bw_out), axis=0)
        blstm_out = lstm_fw_out + lstm_bw_out
        blstm_out, blstm_out_shape = self._to_2d(blstm_out)
        ff_1 = chainer_func.clipped_relu(
            self.ff_1(chainer_func.dropout(blstm_out, dropout, train)))
        ff_2 = chainer_func.clipped_relu(
            self.ff_2(chainer_func.dropout(ff_1, dropout, train)))
        ff_3 = chainer_func.clipped_relu(
            self.ff_3(chainer_func.dropout(ff_2, dropout, train)))
        mask_X = self._to_3d(chainer_func.sigmoid(self.ff_X_mask(ff_3)),
                             blstm_out_shape)
        mask_N = self._to_3d(chainer_func.sigmoid(self.ff_X_mask(ff_3)),
                             blstm_out_shape)
        return mask_N, mask_X
