# DeepClustering_for_Speech_Separation
## TRAINING:
1. Setting config
        When you train for the first time, ['train']['resume']['state'] must be 'False'

2. Edit train_run.sh
> python train.py ./config.yaml ./training_wav_folder

3. Run train_run.sh
>`$ sh train_run.sh`

## TEST:
1. Edit test_run.sh
> python test.py /test_wav_folder ./checkpoint/model.pt mask_option(=hard, soft) eval_idx(=SDR,SI-SDR)
2. Run test_run.sh
>`$ sh test_run.sh`