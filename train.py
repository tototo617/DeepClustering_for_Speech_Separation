import sys
import torch
import yaml
import create_scp
import calc_normalize_params
import sys
from Learning import model
from Learning.trainer import Trainer
from Learning.dataloader import make_dataloader
import datetime




def train():
    path_config = sys.argv[1]
    path_wav_train = sys.argv[2]

    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
        num_spks = config['num_spks']

    config['path_wav_train'] = path_wav_train
        
    dpcl = model.DeepClustering(config)

    dt_now = datetime.datetime.now()

    time = str(dt_now.isoformat())
    print(time)

    create_scp.train_scp(path_wav_train,num_spks)
    calc_normalize_params.dump_dict(config,time)
    

    path_model = "./checkpoint/DeepClustering_config/"+time+"/xx.pt"

    path_scp_mix_tr = "./scp/tr_mix.scp"
    path_scp_targets_tr = ["./scp/tr_s{0}.scp".format(str(i+1)) for i in range(config["num_spks"])]

    path_scp_mix_cv = "./scp/cv_mix.scp"
    path_scp_targets_cv = ["./scp/cv_s{0}.scp".format(str(i+1)) for i in range(config["num_spks"])]

    train_dataloader = make_dataloader(config, path_scp_mix_tr, path_scp_targets_tr, path_model)
    valid_dataloader = make_dataloader(config, path_scp_mix_cv, path_scp_targets_cv, path_model)

    trainer = Trainer(dpcl, config, time)

    trainer.run(train_dataloader, valid_dataloader)



if __name__ == "__main__":
    train()