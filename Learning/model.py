import torch
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pack_sequence,pad_packed_sequence

class DeepClustering(nn.Module):
    def __init__(self,config):
        super().__init__()
        n_fft = config['transform']['n_fft']
        input_size = int(n_fft/2 + 1)
        hidden_size = config['network']['hidden_size']
        num_layers = config['network']['num_layer']
        emb_D = config['network']['emb_D']
        dropout = config['network']['dropout']
        bidirectional = config['network']['bidirectional']
        activation = config['network']['activation']
        

        self.emb_D = emb_D
        self.blstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = bidirectional)



        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(torch.nn, activation)()
        self.linear = nn.Linear((2*hidden_size if bidirectional else hidden_size), input_size * emb_D)
        self.D = emb_D

    def forward(self, x, is_train=True):
        if not is_train:
            x = torch.unsqueeze(x, 0)
        # B x T x F -> B x T x hidden
        x, _ = self.blstm(x)
        x = self.dropout(x)
        # B x T x hidden -> B x T x FD
        x = self.linear(x)
        x = self.activation(x)

        B = x.shape[0]
        if is_train:
            # B x TF x D
            x = x.view(B,-1,self.D)
        else:
            # B x TF x D -> TF x D
            x = x.view(-1, self.D)

        return x


def loss(embs_mix,non_sil,masks,num_spks,device):
    '''
    mix_wave: B x TF x D
    target_waves: B x T x F
    non_silent: B x T x F 
    '''
    B, T, F = non_sil.shape
    # B x TF x spks
    target_embs = masks
    # target_embs = target_embs.to(torch.device(device))

    # B x TF x 1
    non_sil = non_sil.view(B, T*F, 1)

    embs_mix = embs_mix * non_sil
    target_embs = target_embs * non_sil

    vTv = torch.norm(torch.bmm(torch.transpose(embs_mix,1,2),embs_mix),p=2)**2
    vTy = torch.norm(torch.bmm(torch.transpose(embs_mix,1,2),target_embs),p=2)**2
    yTy = torch.norm(torch.bmm(torch.transpose(target_embs,1,2),target_embs),p=2)**2

    loss_embs = (vTv - 2*vTy + yTy)/torch.sum(non_sil)

    return loss_embs

if __name__ == "__main__":
    pass