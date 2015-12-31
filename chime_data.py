import json
import os
import pickle

import numpy as np
import tqdm

from fgnt.mask_estimation import estimate_IBM
from fgnt.signal_processing import audioread
from fgnt.signal_processing import stft
from fgnt.utils import mkdir_p


def _gen_flist_wo_ext(chime_data_dir, stage, scenario):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, scenario))) as fid:
        annotations = json.load(fid)
    flist = [os.path.join(
        chime_data_dir, 'audio', '16kHz', 'isolated_ext',
        '{}05_{}_{}'.format(stage, a['environment'].lower(), scenario),
        '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
             for a in annotations]
    return flist


def _get_audio_data(file_template, postfix='', ch_range=range(1, 7)):
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
            file_template + '.CH{}{}.wav'.format(ch, postfix))[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data


def prepare_training_data(chime_data_dir, dest_dir):
    for stage in ['tr', 'dt']:
        flist = _gen_flist_wo_ext(chime_data_dir, stage, 'simu')
        export_flist = list()
        mkdir_p(os.path.join(dest_dir, stage))
        for f in tqdm.tqdm(flist, desc='Generating data for {}'.format(stage)):
            clean_audio = _get_audio_data(f, '.Clean')
            noise_audio = _get_audio_data(f, '.Noise')
            X = stft(clean_audio, time_dim=1).transpose((1, 0, 2))
            N = stft(noise_audio, time_dim=1).transpose((1, 0, 2))
            IBM_X, IBM_N = estimate_IBM(X, N)
            Y_abs = np.abs(X + N)
            export_dict = {
                'IBM_X': IBM_X.astype(np.float32),
                'IBM_N': IBM_N.astype(np.float32),
                'Y_abs': Y_abs.astype(np.float32)
            }
            export_name = os.path.join(dest_dir, stage, f.split('/')[-1])
            with open(export_name, 'wb') as fid:
                pickle.dump(export_dict, fid)
            export_flist.append(os.path.join(stage, f.split('/')[-1]))
        with open(os.path.join(dest_dir, 'flist_{}.json'.format(stage)),
                  'w') as fid:
            json.dump(export_flist, fid, indent=4)
