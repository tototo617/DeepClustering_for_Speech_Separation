
tensorboard --logdir ./tbx --port 6006 &

path_traindata[0]="/data1/h_munakata/dataset/grid_remove_0.25_0.7"
# path_traindata[1]="/data1/h_munakata/dataset/grid_remove_0.35_0.7"
# path_traindata[2]="/data1/h_munakata/dataset/grid_remove_0.45_0.7"
# path_traindata[3]="/data1/h_munakata/dataset/grid_remove_0.25_1.3"
# path_traindata[4]="/data1/h_munakata/dataset/grid_remove_0.35_1.3"
# path_traindata[5]="/data1/h_munakata/dataset/grid_remove_0.45_1.3"
# path_traindata[6]="/data1/h_munakata/dataset/grid_remove_0.25_1.9"
# path_traindata[7]="/data1/h_munakata/dataset/grid_remove_0.35_1.9"
# path_traindata[8]="/data1/h_munakata/dataset/grid_remove_0.45_1.9"
for i in 0;do
    python train.py ./config.yaml ${path_traindata[i]}
done
# python train.py ./config.yaml /data1/h_munakata/removed