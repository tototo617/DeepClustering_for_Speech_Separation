import os
import sys


def save_scp(dir_wav,scp_name):
    os.makedirs("./scp", exist_ok=True)
    path_scp = "./scp" + "/" + scp_name

    if not os.path.exists(dir_wav):
        raise ValueError("directory of .wav doesn't exist")

    with open(path_scp,'w') as scp:
        for root, dirs, files in os.walk(dir_wav):
            files.sort()
            for file in files:
                scp.write(file+" "+root+'/'+file)
                scp.write('\n')
            



if __name__=="__main__":

    dir_dataset = sys.argv[1]
    train_test = sys.argv[2]

    if train_test=="train" :
        type_list = ['/tr','/cv']
    
    elif train_test=="test":
        type_list = ['tt']

    else:
        raise ValueError("inappropriate data type. try \"train\" or \"test\"")

    print('making scp files')

    for type_data in type_list:
        print("{0} into {1} scp".format(dir_dataset,train_test))
        for i in range(1,3):
            dir_wav = dir_dataset + type_data + '/s' + str(i)
            scp_name = type_data + '_s' + str(i) + '.scp'
            save_scp(dir_wav,scp_name)