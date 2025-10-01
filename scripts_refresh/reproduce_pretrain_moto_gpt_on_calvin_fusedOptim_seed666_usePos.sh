export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd ${PROJECT_ROOT}/moto_gpt/train
accelerate launch --main_process_port 29501 train_moto_gpt.py --config_path "${PROJECT_ROOT}/moto_gpt/configs/train_refresh/${CONFIG_NAME}.yaml"



<<COMMENT
conda activate moto
export PROJECT_ROOT=/dataset_rc_mm/cheny117@xiaopeng.com/Moto
export CONFIG_NAME="reproduce_data_calvin_no_repeat-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb1.0_usePos-train_lr0.0001_bs512_accu4-aug_shiftTrue_resizedCropFalse_fusedOptim_seed666"
ps aux | grep ${CONFIG_NAME} | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts_refresh/
nohup bash reproduce_pretrain_moto_gpt_on_calvin_fusedOptim_seed666_usePos.sh > reproduce_pretrain_moto_gpt_on_calvin_fusedOptim_seed666_usePos.log 2>&1 &
tail -f reproduce_pretrain_moto_gpt_on_calvin_fusedOptim_seed666_usePos.log
COMMENT