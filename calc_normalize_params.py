import numpy as np
import yaml
from tqdm import tqdm
import pickle
from Learning import utils
import os


def calc_normalize_params(path_scp_target_i,config,time):
    wp = utils.wav_processor(config)
    scp_mix = wp.read_scp(path_scp_target_i)

    f_bin = int(config['transform']['n_fft']/2+1)

    mean_f = np.zeros(f_bin)
    var_f = np.zeros(f_bin)

    for key in tqdm(scp_mix.keys()):
        y = wp.read_wav(scp_mix[key])
        logpow = wp.log_power(y,normalize=False)

        mean_f += np.mean(logpow, 0)
        var_f += np.mean(logpow**2, 0)

    mean_f = mean_f / len(scp_mix.keys())
    var_f = var_f / len(scp_mix.keys())
    std_f = np.sqrt(var_f - mean_f**2)

    return mean_f, std_f

def dump_dict(config,time):
    print('calc normalizing parameters')
    num_spks = config['num_spks']
    path_scp_targets = ["./scp/tr_s{0}.scp".format(str(i+1)) for i in range(num_spks)]
    
    total_mean_f, total_std_f = (0,0)
    for path_scp_target_i in path_scp_targets:
        mean_f, std_f = calc_normalize_params(path_scp_target_i,config,time)
        total_mean_f += mean_f
        total_std_f += std_f

    total_mean_f,total_std_f = (total_mean_f/num_spks, total_std_f/num_spks)
    

    path_model = os.path.join("./checkpoint/DeepClustering_config",time)
    os.makedirs(path_model ,exist_ok=True)
    path_normalize = path_model +'/dict_normalize.ark'
    print(path_normalize)


    with open(path_normalize, "wb") as f:
        normalize_dict = {"mean": mean_f, "std": std_f}
        pickle.dump(normalize_dict, f)
    print("Global mean: {}".format(mean_f))
    print("Global std: {}".format(std_f))




if __name__=="__main__":
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    dump_dict(config)