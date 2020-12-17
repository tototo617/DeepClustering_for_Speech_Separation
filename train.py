import sys
import torch
import yaml
import create_scp
import calc_normalize_params
import sys
from Learning import model
from Learning.trainer import Trainer
from Learning.dataloader import make_dataloader




def train():
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    dpcl = model.DeepClustering(config)

    num_spks = config['num_spks']
    path_wav_train = sys.argv[1]
    create_scp.train_scp(path_wav_train,num_spks)
    calc_normalize_params.dump_dict(config)
    

    if config['train']['resume']['state']: 
        path_model = config['train']['resume']['path']
    else:
        path_model = "./checkpoint/DeepClustering_config/xx.pt"

    path_scp_mix_tr = "./scp/tr_mix.scp"
    path_scp_targets_tr = ["./scp/tr_s{0}.scp".format(str(i+1)) for i in range(config["num_spks"])]

    path_scp_mix_cv = "./scp/cv_mix.scp"
    path_scp_targets_cv = ["./scp/cv_s{0}.scp".format(str(i+1)) for i in range(config["num_spks"])]

    train_dataloader = make_dataloader(config, path_scp_mix_tr, path_scp_targets_tr, path_model)
    valid_dataloader = make_dataloader(config, path_scp_mix_cv, path_scp_targets_cv, path_model)

    trainer = Trainer(dpcl, config)

    del dpcl

    trainer.run(train_dataloader, valid_dataloader)


if __name__ == "__main__":
    train()