import os
import yaml
import librosa
import soundfile as sf
import numpy as np
import pickle
import matplotlib.pyplot as plt

class wav_processor:
    def __init__(self,config,path_model=None):
        self.n_fft = config['transform']['n_fft']
        self.hop_length = config['transform']['hop_length']
        self.win_length = config['transform']['win_length']
        self.window = config['transform']['window']
        self.center = config['transform']['center']
        self.sr = config['transform']['sr']
        self.mask_threshold = config['transform']['mask_threshold']

        if path_model!=None:
            path_normalize = os.path.join(os.path.dirname(path_model),'dict_normalize.ark')
            if os.path.exists(path_normalize):
                self.normalize = pickle.load(open(path_normalize, 'rb'))

    def stft(self,y):
        Y = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            center=self.center)
        return Y.T


    def istft(self, Y):
        Y = Y.T
        y = librosa.istft(Y, hop_length=self.hop_length,win_length=self.win_length,
                            window=self.window,center=self.center)
        return y

    def power(self,y):
        Y = self.stft(y)
        eps = np.finfo(float).eps
        return np.maximum(np.abs(Y),eps)


    def log_power(self,y,normalize=True):
        power = self.power(y)
        log_power =  np.log(power)
        if normalize:
            return (log_power - self.normalize['mean']) / self.normalize['std']
        else:
            return log_power


    def non_silent(self,y):
        Y = self.stft(y)
        eps = np.finfo(float).eps
        Y_db = 20 * np.log10(np.maximum(np.abs(Y),eps))
        max_db = np.max(Y_db)
        threshold_magnitude = 10**((max_db - self.mask_threshold) / 20)
        non_silent = np.array(Y > threshold_magnitude, dtype=np.float32)
        return non_silent


    def read_wav(self,wav_path):
        y,_ = sf.read(wav_path)
        return y


    def write_wav(self,dir_path, filename, y):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)
        sf.write(file_path, y, self.sr)


    def read_scp(self,scp_path):
        files = open(scp_path, 'r')
        lines = files.readlines()
        scp_wav = {}
        for line in lines:
            line = line.split()
            if line[0] in scp_wav.keys():
                raise ValueError
            scp_wav[line[0]] = line[1]
        return scp_wav


    def calc_sdr(self,Y,Y_hat):
        Y_sum = np.sum(np.sum((np.abs(Y))**2,1),0)
        Y_hat_sum = np.sum(np.sum((np.abs(Y_hat))**2,1),0)
        
        c = (Y_sum/Y_hat_sum)**(0.5)

        dist = np.sum(np.sum((np.abs(Y) - c*np.abs(Y_hat))**2,1),0)
        sdr = 10*np.log10(Y_sum/dist)

        return sdr


    def calc_SI_SDR(self,Y,Y_hat):
        y = self.istft(Y)
        y_hat = self.istft(Y_hat)
        
        alpha = np.inner(y,y_hat)/np.inner(y,y)

        e_target = np.inner(alpha*y,alpha*y)
        e_res = np.inner(alpha*y - y_hat ,alpha*y - y_hat)

        si_sdr = 10*np.log10(e_target/e_res)
        return si_sdr

    
    def eval_sdr(self,Y_list,Y_hat_list):
        num_spks = len(Y_list)

        sdr = []
        for i in range(num_spks):
            sdr_i = -np.inf
            for j in range(num_spks):
                sdr_j = self.calc_sdr(Y_list[i], Y_hat_list[j])
                if sdr_j > sdr_i:
                    sdr_i = sdr_j
            sdr.append(sdr_i)

        return sdr


    def eval_SI_SDR(self,Y_list,Y_hat_list,Y_mix):
        num_spks = len(Y_list)

        SI_SDR = []
        for i in range(num_spks):
            SI_SDR_max = -np.inf
            for j in range(num_spks):
                SI_SDR_j = self.calc_SI_SDR(Y_list[i], Y_hat_list[j])
                if SI_SDR_max < SI_SDR_j:
                    SI_SDR_max = SI_SDR_j
            SI_SDR.append(SI_SDR_max)

        SI_SDRi = []

        for i in range(num_spks):
            SI_SDR_base = self.calc_SI_SDR(Y_list[i], Y_mix)
            SI_SDRi.append(SI_SDR[i] - SI_SDR_base)
        
        return SI_SDR,SI_SDRi


if __name__ == "__main__":
    with open('../config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    path_model = "./checkpoint/DeepClustering_config/best.pt"
    wp = wav_processor(config,path_model)


