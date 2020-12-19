import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from Learning import model,utils
from Learning.dataloader import transform,padding
import create_scp
import numpy as np
import os
import yaml
from tqdm import tqdm
import datetime
import csv
import sys


class Separation():
    def __init__(self, config, path_wav_test, path_model, clustering_type, eval_idx=None):
        print(eval_idx)

        self.wp = utils.wav_processor(config,path_model)
        self.model = model.DeepClustering(config)
        self.device = torch.device(config['gpu'])
        print('Processing on',config['gpu'])

        self.path_model = path_model
        ckp = torch.load(self.path_model,map_location=self.device)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.model.eval()

        self.trans = transform(config,path_model)

        self.num_spks = config['num_spks']
        self.kmeans = KMeans(n_clusters=self.num_spks)
        self.gmm = GMM(n_components=self.num_spks, max_iter=100)
        dt_now = datetime.datetime.now()
        self.path_separated = "./result_test" + '/'+str(dt_now.isoformat())
        self.clustering_type = clustering_type

        self.scp_mix = self.wp.read_scp("./scp/tt_mix.scp")

        os.makedirs(self.path_separated,exist_ok=True)
        self.path_csv = os.path.join(self.path_separated,"log.csv")
        self.eval_idx = eval_idx if eval_idx!=None else None

        with open(self.path_csv, 'w') as f:
            writer = csv.writer(f)
            conditions = [self.path_model, path_wav_test ,self.eval_idx]
            writer.writerow(conditions)

            header = ["key"]
            if self.eval_idx == 'SDR':
                for i in range(self.num_spks):
                    header.append("SDR_{0}".format(str(i+1)))
                for i in range(self.num_spks):
                    header.append("SDRi_{0}".format(str(i+1)))
            elif self.eval_idx == 'SI-SDR':
                for i in range(self.num_spks):
                    header.append("SI-SDR_{0}".format(str(i+1)))
                for i in range(self.num_spks):
                    header.append("SI-SDRi_{0}".format(str(i+1)))
            writer.writerow(header)



    def est_mask(self, wave, non_silent):
        '''
            input: T x F
        '''
        # TF x D


        mix_emb = self.model(torch.tensor(
            wave, dtype=torch.float32), is_train=False)
        mix_emb = mix_emb.detach().numpy()
        # N x D
        T, F = non_silent.shape
        non_silent = non_silent.reshape(-1)
        # print(non_silent)
        # mix_emb = (mix_emb.T*non_silent).T
        # N
        targets_mask = []

        # hard clustering
        if self.clustering_type == 'hard':
            mix_cluster = self.kmeans.fit_predict(mix_emb)
            for i in range(self.num_spks):
                mask = (mix_cluster == i)
                mask = mask.reshape(T,F)
                targets_mask.append(mask)

        # soft clustering
        elif self.clustering_type == 'soft':
            self.gmm.fit(mix_emb)
            mix_cluster_soft = self.gmm.predict_proba(mix_emb)
            for i in range(self.num_spks):
                mask = mix_cluster_soft[:,i]
                mask = mask.reshape(T,F)
                targets_mask.append(mask)

        return targets_mask


    def eval_separate(self,key,Y_separated,Y_targets,Y_mix):
        if self.eval_idx == 'SDR':
            SDR = self.wp.eval_SDR(Y_targets,Y_separated)
            with open(self.path_csv, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([key]+SDR,SI+SDRi)
        elif self.eval_idx == 'SI-SDR':
            SI_SDR,SI_SDRi = self.wp.eval_SI_SDR(Y_targets,Y_separated,Y_mix)

            with open(self.path_csv, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([key]+SI_SDR+SI_SDRi)


    def run(self):
        if self.eval_idx!=None:
            scp_targets = [self.wp.read_scp("./scp/tt_s{0}.scp".format(str(i+1))) for i in range(self.num_spks)]

        for key in tqdm(self.scp_mix.keys()):
            y_mix = self.wp.read_wav(self.scp_mix[key])
            Y_mix = self.wp.stft(y_mix)

            logpow_mix, non_sil = self.trans(y_mix)
            target_mask = self.est_mask(logpow_mix, non_sil)

            Y_separated = []
            for i in range(len(target_mask)):
                Y_separated.append(target_mask[i] * Y_mix)

            if self.eval_idx!=None:
                y_targets = [self.wp.read_wav(scp_target[key]) for scp_target in scp_targets]
                Y_targets = [self.wp.stft(y_target) for y_target in y_targets]

                self.eval_separate(key,Y_separated,Y_targets,Y_mix)


            for i,Y_separated_i in enumerate(Y_separated):
                y_separated_i = self.wp.istft(Y_separated_i)
                self.wp.write_wav(self.path_separated+'/separated',key.replace('.wav','') + '_'
                                    + str(i+1) + '.wav',y_separated_i)




if __name__ == "__main__":
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
        num_spks = config['num_spks']

    path_separated = "./result_test"
    path_wav_test = sys.argv[1]
    path_model = sys.argv[2]
    clustering_type = sys.argv[3] # "hard" or "soft"
    eval_idx = sys.argv[4] # "SI-SDR" or "SDR"

    create_scp.test_scp(path_wav_test,num_spks)

    separation = Separation(config, path_wav_test, path_model, clustering_type,eval_idx)
    separation.run()