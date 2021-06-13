#!/usr/bin/env bash
SAVE_MODEL_PATH=./model_cate_only
#change to the allennlp 0.9.0 path on your device
ALLENNLP_PATH=/home/ruolin/anaconda3/envs/allen090/bin/allennlp 
rm -rf ${SAVE_MODEL_PATH}
python ${ALLENNLP_PATH} train config_nosp_act.jsonnet -s ${SAVE_MODEL_PATH} --include-package act_aware_dst