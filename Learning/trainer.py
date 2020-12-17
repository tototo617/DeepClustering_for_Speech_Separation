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
    def __init__(self,train_dataloader,val_dataloader,dpcl,config):
        self.cur_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.config = config
        self.num_spks = config['num_spks']
        self.total_epoch = config['train']['epoch']
        self.early_stop = config['train']['early_stop']
        self.checkpoint = config['train']['path']
        self.name = config['name']

        # setting about optimizer
        opt_name = config['optim']['name']
        weight_decay = config['optim']['weight_decay']
        lr = config['optim']['lr']
        momentum = config['optim']['momentum']

        optimizer = getattr(torch.optim, opt_name)
        if opt_name == 'Adam':
            self.optimizer = optimizer(dpcl.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer(dpcl.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        
        if config['optim']['clip_norm']:
            self.clip_norm = config['optim']['clip_norm']
        else:
            self.clip_norm = 0

        # setting about machine

        self.device = torch.device(config['gpu'])
        self.dpcl = dpcl.to(self.device)
        print('Load on:',config['gpu'])
        if config['multi_gpu']:
            self.dpcl = torch.nn.DataParallel(dpcl)
            print('Using multi GPU')

        
        
        # setting about restart
        if config['resume']['state']:    
            ckp = torch.load(config['resume']['path'])
            self.cur_epoch = ckp['epoch']
            self.dpcl.load_state_dict(ckp['model_state_dict'])
            self.optimizer.load_state_dict(ckp['optim_state_dict'])

            print('training resume epoch:',self.cur_epoch)
        

    def train(self, epoch):
        self.dpcl.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0

        for log_pow_mix, class_targets, non_silent in tqdm(self.train_dataloader):
            log_pow_mix = log_pow_mix.to(self.device)
            class_targets = class_targets.to(self.device)
            non_silent = non_silent.to(self.device)

            embs_mix = self.dpcl(log_pow_mix)
            epoch_loss = loss(embs_mix, class_targets, non_silent, self.num_spks, self.device)
            total_loss += epoch_loss.item()

            self.optimizer.zero_grad()
            epoch_loss.backward()
            
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.dpcl.parameters(),self.clip_norm)

            self.optimizer.step()

        total_loss = total_loss/num_batchs

        return total_loss

    def validation(self, epoch):
        self.dpcl.eval()
        num_batchs = len(self.val_dataloader)
        total_loss = 0.0
        with torch.no_grad():
            for log_pow_mix, class_targets, non_silent in tqdm(self.val_dataloader):
                log_pow_mix = log_pow_mix.to(self.device)
                class_targets = class_targets.to(self.device)
                non_silent = non_silent.to(self.device)

                embs_mix = self.dpcl(log_pow_mix)

                epoch_loss = loss(embs_mix, class_targets, non_silent, self.num_spks, self.device)
                total_loss += epoch_loss.item()
    
        total_loss = total_loss/num_batchs

        return total_loss
    
    def run(self):
        train_loss = []
        val_loss = []
        print('cur_epoch',self.cur_epoch)

        dt_now = datetime.datetime.now()
        writer = tbx.SummaryWriter("tbx/" + dt_now.isoformat())
        os.makedirs('./checkpoint/DeepClustering_config',exist_ok=True)
        logging.basicConfig(filename='./checkpoint/DeepClustering_config/train_log.log', level=logging.DEBUG)
        logging.info(self.config)
        with torch.cuda.device(self.device):
            self.save_checkpoint(self.cur_epoch,best=False)
            v_loss = self.validation(self.cur_epoch)
            best_loss = v_loss
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(self.cur_epoch)
                logging.info('epoch{0}:train_loss{1}'.format(self.cur_epoch,t_loss))
                print('epoch{0}:train_loss{1}'.format(self.cur_epoch,t_loss))
                v_loss = self.validation(self.cur_epoch)
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
        print('save model epoch:',epoch)
        os.makedirs(os.path.join(self.checkpoint,self.name),exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.dpcl.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
        os.path.join(self.checkpoint,self.name,'{0}.pt'.format('best' if best else 'last')))