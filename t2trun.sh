#! /bin/zsh
#conda activate ddd
nohup python -u ./src/model/t2t_model.py >> ./log/t2t.log 2>&1 &