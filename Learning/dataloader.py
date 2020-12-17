import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from Learning import utils
import yaml
import matplotlib.pyplot as plt


class transform():
    def __init__(self,config,path_model):
        self.num_spks = config['num_spks']
        self.wp = utils.wav_processor(config,path_model)
        self.mask_type = config['train']['mask_type']

    def __call__(self,y_mix,y_targets=None):
        logpow_mix = self.wp.log_power(y_mix)
        non_sil = self.wp.non_silent(y_mix)

        if y_targets:
            masks = self.make_mask(y_targets,non_sil)
            return logpow_mix, non_sil, masks
        else:
            return logpow_mix, non_sil

    def make_mask(self,y_targets,non_sil):
        pow_targets = [self.wp.power(y_target) for y_target in y_targets]
        T,F = pow_targets[0].shape

        masks = np.zeros([T*F, self.num_spks])

        if self.mask_type=='IBM':
            mask_targets = np.argmax(np.array(pow_targets), axis=0)
            for i in range(self.num_spks):
                mask_i = np.ones(non_sil.shape) * (mask_targets==i)
                masks[:,i] = mask_i.reshape([T*F])

        return masks

        # elif self.mask=='IRM':
        #     eps = np.finfo(np.float64).eps
        #     sum_pow_targets = sum(pow_targets)
            
        #     for i,pow_target in enumerate(pow_targets):
        #         class_targets[:,i] = (pow_target / (sum_pow_targets + eps)).reshape([T*F])
        #     print(class_targets)
        
        # elif self.mask=='WM':
        #     eps = np.finfo(np.float64).eps
        #     pow_targets = [np.power(pow_target,2) for pow_target in pow_targets]
        #     sum_pow_targets = sum(pow_targets)

        #     for i,pow_target in enumerate(pow_targets):
        #         class_targets[:,i] = (pow_target / (sum_pow_targets + eps)).reshape([T*F])


class wav_dataset(Dataset):
    def __init__(self,config,path_scp_mix,path_scp_targets,path_model):
        self.wp = utils.wav_processor(config,path_model)

        self.scp_mix = self.wp.read_scp(path_scp_mix)
        self.scp_targets = [self.wp.read_scp(path_scp_target_i) \
                                for path_scp_target_i in path_scp_targets]

        self.keys = [key for key in self.scp_mix.keys()]

        self.trans = transform(config,path_model)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        y_mix = self.wp.read_wav(self.scp_mix[key])
        y_targets = [self.wp.read_wav(scp_target_i[key]) \
                                for scp_target_i in self.scp_targets]
        
        return self.trans(y_mix,y_targets)


def padding(batch):
    batch_log_pow_mix,batch_class_targets,batch_non_silent = [],[],[]
    for log_pow_mix,class_targets,non_silent in batch:
        batch_log_pow_mix.append(torch.tensor(log_pow_mix,dtype=torch.float32))
        batch_class_targets.append(torch.tensor(class_targets,dtype=torch.int64))
        batch_non_silent.append(torch.tensor(non_silent,dtype=torch.float32))

    batch_log_pow_mix = pad_sequence(batch_log_pow_mix, batch_first=True)
    batch_class_targets = pad_sequence(batch_class_targets, batch_first=True)
    batch_non_silent = pad_sequence(batch_non_silent, batch_first=True)

    return batch_log_pow_mix,batch_class_targets,batch_non_silent


def make_dataloader(config, path_scp_mix, path_scp_targets, path_model):
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    shuffle = config['dataloader']['shuffle']
    dataset  = wav_dataset(config, path_scp_mix, path_scp_targets,path_model)

    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,
                                shuffle=shuffle,collate_fn=padding)

    return dataloader


if __name__ == "__main__":
    with open('../config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    path_model = "../checkpoint/DeepClustering_config/best.pt"
    path_scp_mix = "../scp/tt_mix.scp"

    path_scp_targets = ["../scp/tt_s1.scp",
                        "../scp/tt_s2.scp"]

    dataloader = make_dataloader(config, path_scp_mix, path_scp_targets, path_model)

    for i,(b_logpow,b_nonsil,b_masks) in enumerate(dataloader):
        if i==0:
            print("logpow",b_logpow[0,:,:])
            print("nonsil",b_nonsil[0,:,:])
            print("masks",b_masks[0,:,:])

