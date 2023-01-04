#! /bin/zsh
#conda activate ddd
#nohup python -u ./src/main_run.py --datadir='./dataset/Wiki10-31K/' --istrain=1 --is_pred_trn=1 --is_pred_tst=1 --iscombine=1 --is_rank_train=1 --is_ranking=1 --combine_model='cross-encoder/stsb-roberta-base' --modelname='t5' --outputmodel='t5_save' --batch_size=8 --epoch=10 --checkdir='t5_check' --data_size=4  --rank_model='all-MiniLM-L6-v2' --rank_batch=128 --rankmodel_save='bi_en_t5'>> ./log/output.log 2>&1 &
nohup python -u ./src/main_run.py --datadir='./dataset/Wiki10-31K/' \
--istrain=1 \
--is_pred_trn=1 \
--is_pred_tst=1 \
--iscombine=0 \
--is_rank_train=1  \
--is_ranking=1 \
--combine_model='bi-encoder' \
--modelname='Pegasus' \
--outputmodel='pegasus_save' \
--batch_size=2 \
--epoch=5 \
--checkdir='pega_check' \
--data_size=4  \
--rank_model='all-MiniLM-L6-v2' \
--rank_batch=32 \
--rankmodel_save='bi_en_' \
>> ./log/output.log 2>&1 &
