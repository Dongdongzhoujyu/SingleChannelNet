# SingleChannelNet #


Code for the model in the paper SingleChannelNet: A model for automatic sleep stage classification with raw single-channel EEG by Dongdong Zhou, Jian Wang, Guoqiang Hu, Jiacheng Zhang, Fan Li, Rui Yan, Lauri Kettunen, Zheng Chang, Qi Xu, Fengyu Cong.

This work has been accepted for publication in [Biomedical signal processing and control](https://www.sciencedirect.com/science/article/pii/S1746809422001148).

The structure of SingleChannelNet is shown as:

![SingleChannelNet](./images/SingleChannelNet.png)


## Environment ##

- Ubuntu 18.04 / Windows 10 1903 x64
- CUDA toolkit 11.6 and CuDNN v7.6.4
- Python 3.6.7
- tensorflow-gpu (1.12.0)
- Keras (2.2.4)
- matplotlib (3.2.2)
- scikit-learn (0.23.1)
- scipy (1.5.0)
- numpy (1.16.0)
- pandas (1.1.0)
- mne (0.21.2)
- h5py (2.10.0)



## Evaluation datasets ##
We evaluated our SingelChannelNet with [CCSHS](https://sleepdata.org/datasets/ccshs) and [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) datasets.


Then run the following script to extract specified EEG channels and their corresponding sleep stages.

    python prepare_physionet.py --data_dir data --output_dir data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
    python prepare_physionet.py --data_dir data --output_dir data/eeg_pz_oz --select_ch 'EEG Pz-Oz'


## Summary ##
Run this script to show a summary of the performance of our DeepSleepNet compared with the state-of-the-art hand-engineering approaches. The performance metrics are overall accuracy, per-class F1-score, and macro F1-score.

    python summary.py --data_dir output
    

## Citation ##
If you find this useful, please cite our work as follows:

        @article{zhou2022singlechannelnet,
          title={Singlechannelnet: A model for automatic sleep stage classification with raw single-channel eeg},
          author={Zhou, Dongdong and Wang, Jian and Hu, Guoqiang and Zhang, Jiacheng and Li, Fan and Yan, Rui and Kettunen, Lauri and Chang, Zheng and Xu, Qi and Cong, Fengyu},
          journal={Biomedical Signal Processing and Control},
          volume={75},
          pages={103592},
          year={2022},
          publisher={Elsevier}
        }
## Licence ##
- For academic and non-commercial use only
- Apache License 2.0
