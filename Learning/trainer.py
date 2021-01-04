import sys
from Learning.model import loss
import torch
import os
import yaml
from tqdm import tqdm
import logging
import tensorboardX as tbx
import datetime

class Trainer():
    def __init__(self,model,config,time):
        self.model = model
        self.cur_epoch = 0
        self.config = config
        self.num_spks = config['num_spks']
        self.total_epoch = config['train']['epoch']
        self.early_stop = config['train']['early_stop']
        self.checkpoint = config['train']['path']
        self.name = config['name']

        self.time = time

        # setting about optimizer
        opt_name = config['train']['optim']['name']
        weight_decay = config['train']['optim']['weight_decay']
        lr = config['train']['optim']['lr']
        momentum = config['train']['optim']['momentum']

        optimizer = getattr(torch.optim, opt_name)
        if opt_name == 'Adam':
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.clip_norm = config['train']['optim']['clip_norm'] if config['train']['optim']['clip_norm'] else 0

        # setting about machine
        self.device = torch.device(config['gpu'])
        if config['train']['resume']['state']:    
            self.load_checkpoint(config)

        self.model = self.model.to(self.device)

    def train(self, epoch, dataloader, mode):
        if mode=="train":
            self.model.train()
        elif mode=="valid":
            self.model.eval()
        else:
            raise ValueError("inappropriate mode: try mode = \"train\" or \"valid\"")

        num_batchs = len(dataloader)
        total_loss = 0.0

        for logpow_mix, non_sil, masks in tqdm(dataloader):
            logpow_mix = logpow_mix.to(self.device)
            masks = masks.to(self.device)
            non_sil = non_sil.to(self.device)

            embs_mix = self.model(logpow_mix)
            epoch_loss = loss(embs_mix, non_sil, masks, self.num_spks, self.device)
            total_loss += epoch_loss.item()

            if mode=="train":
                self.optimizer.zero_grad()
                epoch_loss.backward()
                if self.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_norm)
                self.optimizer.step()

        total_loss = total_loss/num_batchs

        return total_loss

    
    def run(self,train_dataloader,valid_dataloader):
        train_loss = []
        val_loss = []
        print('cur_epoch',self.cur_epoch)

        writer = tbx.SummaryWriter("tbx/" + self.time)
        os.makedirs('./checkpoint/DeepClustering_config',exist_ok=True)
        logging.basicConfig(filename='./checkpoint/DeepClustering_config/train_log.log', level=logging.DEBUG)
        logging.info(self.config)
        self.save_checkpoint(self.cur_epoch,best=False)
        v_loss = self.train(self.cur_epoch, valid_dataloader, mode="valid")
        best_loss = v_loss
        no_improve = 0
        # starting training part
        while self.cur_epoch < self.total_epoch:
            self.cur_epoch += 1
            t_loss = self.train(self.cur_epoch, train_dataloader, mode="train")
            logging.info('epoch{0}:train_loss{1}'.format(self.cur_epoch,t_loss))
            print('epoch{0}:train_loss{1}'.format(self.cur_epoch,t_loss))
            v_loss = self.train(self.cur_epoch, valid_dataloader,mode="valid")
            logging.info('epoch{0}:valid_loss{1}'.format(self.cur_epoch,v_loss))
            print('epoch{0}:valid_loss{1}'.format(self.cur_epoch,v_loss))

            writer.add_scalar('t_loss', t_loss, self.cur_epoch)
            writer.add_scalar('v_loss', v_loss, self.cur_epoch)

            train_loss.append(t_loss)
            val_loss.append(v_loss)

            if v_loss >= best_loss:
                no_improve += 1
            else:
                best_loss = v_loss
                no_improve = 0
                self.save_checkpoint(self.cur_epoch,best=True)
            
            if no_improve == self.early_stop:
                break
            self.save_checkpoint(self.cur_epoch,best=False)
        
        writer.close()


    def save_checkpoint(self, epoch, best=True):
        self.model.to('cpu')
        print('save model epoch:{0} as {1}'.format(epoch,"best" if best else "last"))
        os.makedirs(os.path.join(self.checkpoint,self.name),exist_ok=True)

        path_save_model = os.path.join(self.checkpoint,self.name,self.time,
                                            '{0}.pt'.format('best' if best else 'last'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
        path_save_model)

        self.model.to(self.device)

        with open(os.path.join(self.checkpoint,self.name,self.time,'config_backup.yaml'),mode='w') as f:
            f.write(yaml.dump(self.config))


    def load_checkpoint(self,config):
        print('load on:',self.device)

        ckp = torch.load(config['train']['resume']['path'],map_location=torch.device('cpu'))
        self.cur_epoch = ckp['epoch']
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optim_state_dict'])

        self.model = self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        print('training resume epoch:',self.cur_epoch)
