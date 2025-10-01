export CUDA_VISIBLE_DEVICES=4,5,6,7
cd ${PROJECT_ROOT}/moto_gpt/train
accelerate launch --main_process_port 29502 train_moto_gpt.py --config_path "${PROJECT_ROOT}/moto_gpt/configs/train_refresh/${CONFIG_NAME}.yaml"



<<COMMENT
conda activate moto
export PROJECT_ROOT=/dataset_rc_mm/cheny117@xiaopeng.com/Moto
export CONFIG_NAME="reproduce_data_calvin-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5_usePos-train_lr0.0002_bs512_fusedOptim-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch5_fusedOptim_seed666"
ps aux | grep ${CONFIG_NAME} | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts_refresh/
nohup bash reproduce_finetune_moto_gpt_on_calvin_from_epoch5_fusedOptim_seed666_usePos.sh > reproduce_finetune_moto_gpt_on_calvin_from_epoch5_fusedOptim_seed666_usePos.log 2>&1 &
tail -f reproduce_finetune_moto_gpt_on_calvin_from_epoch5_fusedOptim_seed666_usePos.log
COMMENT