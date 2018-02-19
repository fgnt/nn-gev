# Neural network based GEV beamformer

## Introduction

This repository contains code to replicate the results for the 3rd CHiME challenge using a *NN-GEV* Beamformer.

## Install

This code requires Python 3 to run (although most parts should be compatible with Python 2.7). Install the necessary modules:

```
pip install chainer
pip install tqdm
pip install SciPy
pip install scikit-learn
pip install librosa
```

## Usage

  1. Extract the speech and noise images for the SimData using the modified Matlab script in CHiME3/tools/simulation
  2. Start the training for the BLSTM model using the GPU with id 0 and the data directory ``data``:
  
      ```
      python train.py --chime_dir=../chime/data --gpu 0 data BLSTM
      ```
      
      This will first create the training data (i.e. the binary mask targets) and then run the training with early stopping. Instead of ``BLSTM`` it is also possible to specify ``FW`` to train a simple feed-forward model.
      
  3. Start the beamforming:
  
    ```
    beamform.sh ../chime/data data/export_BLSTM data/BLSTM_model/best.nnet BLSTM
    ```
    
    This will apply the beamformer to every utterance of the CHiME database and store the resulting audio file in ``data/export_BLSTM``. The model ``data/BLSTM_model/best.nnet`` is used to generate the masks.
    
  4. Start the kaldi baseline using the exported data.

  If you want to use the beamformer with a different database, take a look at ``beamform.py`` and ``chime_data`` and modify it accordingly.

## Results
With the new baseline, you should get the following results:
  
    ```
    local/chime4_calc_wers.sh exp/tri3b_tr05_multi_noisy new_baseline exp/tri3b_tr05_multi_noisy/graph_tgpr_5k
    compute dt05 WER for each location
    -------------------
    best overall dt05 WER 9.77% (language model weight = 12)
    -------------------
    dt05_simu WER: 9.81% (Average), 8.95% (BUS), 11.28% (CAFE), 8.55% (PEDESTRIAN), 10.44% (STREET)
    -------------------
    dt05_real WER: 9.73% (Average), 11.67% (BUS), 9.37% (CAFE), 8.41% (PEDESTRIAN), 9.47% (STREET)
    -------------------
    et05_simu WER: 10.67% (Average), 8.85% (BUS), 11.34% (CAFE), 11.02% (PEDESTRIAN), 11.47% (STREET)
    -------------------
    et05_real WER: 14.00% (Average), 19.01% (BUS), 13.37% (CAFE), 12.37% (PEDESTRIAN), 11.24% (STREET)
    -------------------


    ./local/chime4_calc_wers_smbr.sh exp/tri4a_dnn_tr05_multi_noisy_smbr_i1lats new_baseline exp/tri4a_dnn_tr05_multi_noisy/graph_tgpr_5k
    compute dt05 WER for each location
    -------------------
    best overall dt05 WER 5.87% (language model weight = 9) (Number of iterations = 4)
    -------------------
    dt05_simu WER: 5.62% (Average), 5.24% (BUS), 6.58% (CAFE), 4.91% (PEDESTRIAN), 5.77% (STREET)
    -------------------
    dt05_real WER: 6.11% (Average), 7.66% (BUS), 5.83% (CAFE), 5.10% (PEDESTRIAN), 5.87% (STREET)
    -------------------
    et05_simu WER: 7.26% (Average), 6.74% (BUS), 7.70% (CAFE), 7.38% (PEDESTRIAN), 7.23% (STREET)
    -------------------
    et05_real WER: 9.48% (Average), 14.06% (BUS), 8.22% (CAFE), 7.81% (PEDESTRIAN), 7.84% (STREET)
    -------------------

    
    local/chime4_calc_wers.sh exp/tri4a_dnn_tr05_multi_noisy_smbr_lmrescore new_baseline_rnnlm_5k_h300_w0.5_n100 exp/tri4a_dnn_tr05_multi_noisy_smbr_lmrescore/graph_tgpr_5k
    compute dt05 WER for each location
    -------------------
    best overall dt05 WER 4.02% (language model weight = 11)
    -------------------
    dt05_simu WER: 3.97% (Average), 3.66% (BUS), 4.65% (CAFE), 3.38% (PEDESTRIAN), 4.19% (STREET)
    -------------------
    dt05_real WER: 4.07% (Average), 5.34% (BUS), 3.61% (CAFE), 3.35% (PEDESTRIAN), 4.00% (STREET)
    -------------------
    et05_simu WER: 4.51% (Average), 4.09% (BUS), 4.61% (CAFE), 4.46% (PEDESTRIAN), 4.86% (STREET)
    -------------------
    et05_real WER: 6.46% (Average), 9.87% (BUS), 5.47% (CAFE), 5.14% (PEDESTRIAN), 5.34% (STREET)
    -------------------
    ```
    
## Citation
  If you use this code for your experiments, please consider citing the following paper:
  
  ```
  @inproceedings{Hey2016,
  title = {NEURAL NETWORK BASED SPECTRAL MASK ESTIMATION FOR ACOUSTIC BEAMFORMING},
  author = {J. Heymann, L. Drude, R. Haeb-Umbach},
  year = {2016},
  date = {2016-03-20},
  booktitle = {Proc. IEEE Intl. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  keywords = {},
  pubstate = {forthcoming},
  tppubtype = {inproceedings}
  }
  ```
