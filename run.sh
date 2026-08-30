# Point this at your local copy of the preprocessed BRATS *_none_npy volumes.
datapath=./data/BRATS2018_Training_none_npy
dataname=BRATS2018

python3 -u train_federated.py --client_num 8 --gpus 0,1,2,3 --c_rounds 1000 --eval 30 --datapath ${datapath} --dataname ${dataname} --setting_options c8 --version brats18_c8 --resume 0
