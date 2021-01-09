path_test_data[0]="/data1/h_munakata/univ_sound/test_100/test_speech_speech"
path_test_data[1]="/data1/h_munakata/univ_sound/test_100/test_speech_whistle"
path_test_data[2]="/data1/h_munakata/univ_sound/test_100/test_speech_phone"
path_test_data[3]="/data1/h_munakata/univ_sound/test_100/test_whistle_whistle"
path_test_data[4]="/data1/h_munakata/univ_sound/test_100/test_whistle_phone"
path_test_data[5]="/data1/h_munakata/univ_sound/test_100/test_phone_phone"


# path_model[0]="./checkpoint/backups/imbalance/best.pt"
# path_model[1]="./checkpoint/backups/balanced/best.pt"
# path_model[2]="./checkpoint/backups/augmented/best.pt"
# path_model[3]="./checkpoint/backups/removed/best.pt"
# path_model[4]="./checkpoint/backups/refine/best.pt"

# path_model[5]="./checkpoint/backups/imbalance2/best.pt"
# path_model[6]="./checkpoint/backups/removed2/best.pt"
# path_model[7]="./checkpoint/backups/refine2/best.pt"

path_model[0]="./checkpoint/DeepClustering_config/grid_remove_0.25_0.7/best.pt"
path_model[1]="./checkpoint/DeepClustering_config/grid_remove_0.35_0.7/best.pt"
path_model[2]="./checkpoint/DeepClustering_config/grid_remove_0.45_0.7/best.pt"
path_model[3]="./checkpoint/DeepClustering_config/grid_remove_0.25_1.3/best.pt"
path_model[4]="./checkpoint/DeepClustering_config/grid_remove_0.35_1.3/best.pt"
path_model[5]="./checkpoint/DeepClustering_config/grid_remove_0.45_1.3/best.pt"

path_model[0]="./checkpoint/DeepClustering_config/teian/best.pt"

mask_type="hard"
eval_idx='SI-SDR'




for i in 0 
do
    for j in 0 1 2 3 4 5
    do
        python test.py ${path_test_data[j]} ${path_model[i]} $mask_type $eval_idx
    done
done