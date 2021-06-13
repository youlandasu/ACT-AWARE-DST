test_file=$1
ALLENNLP_PATH=/home/ruolin/anaconda3/envs/allen090/bin/allennlp
RESPATH=res_act_hybrid
MODELPATH=model_act_hybrid
#rm -i eval_act_sp_withgamma
#touch eval_act_sp_withgamma
rm -r -i ${RESPATH}
mkdir ${RESPATH}
for i in 3 4; do #187 191 195 199
    python ${ALLENNLP_PATH} predict --cuda-device 0 --predictor act_aware_dst --include-package act_aware_dst --weights-file ${MODELPATH}/model_state_epoch_${i}.th ${MODELPATH}/model.tar.gz ${test_file} > ${RESPATH}/${i}
    python formulate_pred_belief_state.py ${RESPATH}/${i} >> eval_act_sp 
done