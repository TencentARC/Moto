<div align="center">
<h1>
Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

<a href='https://chenyi99.github.io/moto/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2412.04445v2'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/TencentARC/Moto'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>
</h1>

![image](assets/teaser.png?raw=true)
 
</div>

## 🚀Introduction

Recent developments in Large Language Models (LLMs) pre-trained on extensive corpora have shown significant success in various natural language processing (NLP) tasks with minimal fine-tuning.
This success offers new promise for robotics, which has long been constrained by the high cost of action-labeled data. We ask: given the abundant video data containing interaction-related knowledge available as a rich "corpus", <b><i>can a similar generative pre-training approach be effectively applied to enhance robot learning?</i></b> The key challenge is to identify an effective representation for autoregressive pre-training that benefits robot manipulation tasks.
Inspired by the way humans learn new skills through observing dynamic environments, we propose that effective robotic learning should emphasize motion-related knowledge, which is closely tied to low-level actions and is hardware-agnostic, facilitating the transfer of learned motions to actual robot actions.

To this end, we introduce <b>Moto</b>, which converts video content into latent <b>Mo</b>tion <b>To</b>ken sequences by a Latent Motion Tokenizer, learning a bridging "language" of motion from videos in an unsupervised manner.
We pre-train Moto-GPT through motion token autoregression, enabling it to capture diverse visual motion knowledge. After pre-training, Moto-GPT demonstrates the promising ability to produce semantically interpretable motion tokens, predict plausible motion trajectories, and assess trajectory rationality through output likelihood.
To transfer learned motion priors to real robot actions, we implement a co-fine-tuning strategy that seamlessly bridges latent motion token prediction and real robot control. Extensive experiments show that the fine-tuned Moto-GPT exhibits superior robustness and efficiency on robot manipulation benchmarks, underscoring its effectiveness in transferring knowledge from video data to downstream visual manipulation tasks.

## 🛠️Quick Start

### Installation
Clone this repo:
```bash
git clone https://github.com/TencentARC/Moto.git
```

Install minimal requirements for Moto training and inference:
```bash
conda create -n moto python=3.8
conda activate moto
cd Moto
pip install -r requirements.txt
cd ..
```


[Optional] Setup the conda environment for evaluating Moto-GPT on the [CALVIN](https://github.com/mees/calvin) benchmark:

```bash
conda create -n moto_for_calvin python=3.8
conda activate moto_for_calvin

git clone --recurse-submodules https://github.com/mees/calvin.git
pip install setuptools==57.5.0
cd calvin
cd calvin_env; git checkout main
cd ../calvin_models
sed -i 's/pytorch-lightning==1.8.6/pytorch-lightning/g' requirements.txt
sed -i 's/torch==1.13.1/torch/g' requirements.txt
cd ..
sh ./install.sh
cd ..

sudo apt-get install -y libegl1-mesa libegl1
sudo apt-get install -y libgl1
sudo apt-get install -y libosmesa6-dev
sudo apt-get install -y patchelf

cd Moto
pip install -r requirements.txt
cd ..
```



[Optional] Setup the conda environment for evaluating Moto-GPT on the [SIMPLER](https://github.com/simpler-env/SimplerEnv) benchmark:
```bash
source /data/miniconda3/bin/activate
conda create -n moto_for_simpler python=3.10 -y
conda activate moto_for_simpler


git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
pip install numpy==1.24.4
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .
cd SimplerEnv
pip install -e .
sudo apt install ffmpeg
pip install setuptools==58.2.0
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1
pip install git+https://github.com/nathanrooy/simulated-annealing
cd ..

cd Moto
pip install -r requirements.txt
cd ..
```

### Model Weights
We release the Latent Motion Tokenizer, the pre-traiend Moto-GPT and the fine-tuned Moto-GPT in [Moto Hugging Face](https://huggingface.co/TencentARC/Moto). You can download them separately and save them in corresponding directories ([`latent_motion_tokenizer/checkpoints/`](latent_motion_tokenizer/checkpoints) and [`moto_gpt/checkpoints/`](moto_gpt/checkpoints)).

## 🤖Inference

### Latent trajectory inference with the pre-trained Moto-GPT and the Latent Motion Tokenizer
```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
cd ${PROJECT_ROOT}/scripts
nohup bash run_latent_motion_generation.sh > run_latent_motion_generation.log 2>&1 &
tail -f run_latent_motion_generation.log
```


### Evaluating the fine-tuned Moto-GPT on robot manipulation benchmarks

 Evaluation on CALVIN
```bash
conda activate moto_for_calvin
export PROJECT_ROOT=[your path to Moto project]
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_moto_gpt_in_calvin.sh > evaluate_moto_gpt_in_calvin.log 2>&1 &
tail -f evaluate_moto_gpt_in_calvin.log
```

Evaluation on SIMPLER
```bash
conda activate moto_for_simpler
export PROJECT_ROOT=[your path to Moto project]
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_moto_gpt_in_simpler.sh > evaluate_moto_gpt_in_simpler.log 2>&1 &
tail -f evaluate_moto_gpt_in_simpler.log
```

## 🔥Training
### Prepare Datasets
#### CALVIN dataset
- Download and preprocess Split ABC->D dataset from [CALVIN](https://github.com/mees/calvin/tree/main/dataset):
```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export OUTPUT_ROOT=[your path to save datasets]
cd ${PROJECT_ROOT}/scripts/
nohup bash download_and_preprocess_calvin_data.sh > download_and_preprocess_calvin_data.log 2>&1 &
tail -f download_and_preprocess_calvin_data.log
```

#### Open X-Embodiment datasets
- Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install)

- Download and preprocess datasets from [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment):
```bash
conda activate moto
pip install tensorflow-datasets
export PROJECT_ROOT=[your path to Moto project]
export OUTPUT_ROOT=[your path to save datasets]
cd ${PROJECT_ROOT}/scripts/
nohup bash download_and_preprocess_oxe_data.sh > download_and_preprocess_oxe_data.log 2>&1 &
tail -f download_and_preprocess_oxe_data.log
```

<!-- - Modify the `video_dir` and `lmdb_dir` fields in data configs from [latent_motion_tokenizer/configs/data/](latent_motion_tokenizer/configs/data/) and [moto_gpt/configs/data/](moto_gpt/configs/data/) -->

### Stage-1: Training Latent Motion Tokenizer
#### Training on CALVIN dataset
- Modify the `npz_dir` field in [latent_motion_tokenizer/configs/data/calvin.yaml](latent_motion_tokenizer/configs/data/calvin.yaml)

- Config the paths in [latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml](latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml)

- Run the following commands:

```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse"
cd ${PROJECT_ROOT}/scripts/
nohup bash train_latent_motion_tokenizer_on_calvin.sh > train_latent_motion_tokenizer_on_calvin.log 2>&1 &
tail -f train_latent_motion_tokenizer_on_calvin.log
```

#### Training on Open X-Embodiment datasets
- Modify the `video_dir` field in [latent_motion_tokenizer/configs/data/rtx.yaml](latent_motion_tokenizer/configs/data/rtx.yaml)

- Config the paths in [latent_motion_tokenizer/configs/train/data_rtx-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.001_bs256-aug_shiftTrue_resizedCropFalse.yaml](latent_motion_tokenizer/configs/train/data_rtx-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.001_bs256-aug_shiftTrue_resizedCropFalse.yaml)

- Run the following commands:

```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_rtx-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.001_bs256-aug_shiftTrue_resizedCropFalse"
cd ${PROJECT_ROOT}/scripts/
nohup bash train_latent_motion_tokenizer_on_oxe.sh > train_latent_motion_tokenizer_on_oxe.log 2>&1 &
tail -f train_latent_motion_tokenizer_on_oxe.log
```



### Stage-2: Pre-training Moto-GPT
#### Pre-training on CALVIN dataset
- Modify the `lmdb_dir` field in [moto_gpt/configs/data/calvin.yaml](moto_gpt/configs/data/calvin.yaml)

- Config the paths in [moto_gpt/configs/train/data_calvin-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0001_bs512-aug_shiftTrue_resizedCropFalse.yaml](moto_gpt/configs/train/data_calvin-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0001_bs512-aug_shiftTrue_resizedCropFalse.yaml)

- Run the following commands:

```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_calvin-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0001_bs512-aug_shiftTrue_resizedCropFalse"
cd ${PROJECT_ROOT}/scripts/
nohup bash pretrain_moto_gpt_on_calvin.sh > pretrain_moto_gpt_on_calvin.log 2>&1 &
tail -f pretrain_moto_gpt_on_calvin.log
```



#### Pre-training on Open X-Embodiment datasets
- Modify the `video_dir` and `lmdb_dir` fields in [moto_gpt/configs/data/rtx.yaml](moto_gpt/configs/data/rtx.yaml)

- Config the paths in [moto_gpt/configs/train/data_rtx-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.001_bs512-aug_shiftTrue_resizedCropFalse.yaml](moto_gpt/configs/train/data_rtx-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.001_bs512-aug_shiftTrue_resizedCropFalse.yaml)

- Run the following commands:

```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_rtx-model_actPredFalse_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.001_bs512-aug_shiftTrue_resizedCropFalse"
ps aux | grep ${CONFIG_NAME} | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts/
nohup bash pretrain_moto_gpt_on_oxe.sh > pretrain_moto_gpt_on_oxe.log 2>&1 &
tail -f pretrain_moto_gpt_on_oxe.log
```


### Stage-3: Fine-tuning Moto-GPT
#### Fine-tuning on CALVIN dataset
- Modify the `lmdb_dir` field in [moto_gpt/configs/data/calvin.yaml](moto_gpt/configs/data/calvin.yaml)

- Config the paths in [moto_gpt/configs/train/data_calvin-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0002_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch10.yaml](moto_gpt/configs/train/data_calvin-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0002_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch10.yaml)

- Run the following commands:

```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_calvin-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0002_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch10"
cd ${PROJECT_ROOT}/scripts/
nohup bash finetune_moto_gpt_on_calvin.sh > finetune_moto_gpt_on_calvin.log 2>&1 &
tail -f finetune_moto_gpt_on_calvin.log
```

#### Fine-tuning on RT-1 dataset
- Modify the `video_dir` and `lmdb_dir` fields in [moto_gpt/configs/data/rt1.yaml](moto_gpt/configs/data/rt1.yaml)

- Config the paths in [moto_gpt/configs/train/data_rt1-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.001_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_oxe_Epoch10.yaml](moto_gpt/configs/train/data_rt1-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.001_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_oxe_Epoch10.yaml)

- Run the following commands:

```bash
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_rt1-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk3_maskProb0.5-train_lr0.001_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_oxe_Epoch10"
cd ${PROJECT_ROOT}/scripts/
nohup bash finetune_moto_gpt_on_rt1.sh > finetune_moto_gpt_on_rt1.log 2>&1 &
tail -f finetune_moto_gpt_on_rt1.log
```

## 📝To Do
- [x] Release the Latent Motion Tokenizer
- [x] Release the pre-trained and fine-tuned Moto-GPT
- [x] Release the inference code
- [x] Release the training code


## 📚Citation
If you find our project helpful, hope you can star our repository and cite our paper as follows:

```
@article{chen2024moto,
  title={Moto: Latent Motion Token as the Bridging Language for Robot Manipulation},
  author={Chen, Yi and Ge, Yuying and Li, Yizhuo and Ge, Yixiao and Ding, Mingyu and Shan, Ying and Liu, Xihui},
  journal={arXiv preprint arXiv:2412.04445},
  year={2024}
}
```

## 🙌Acknowledgement
This repo benefits from [Taming Transformers](https://github.com/CompVis/taming-transformers/), [Phenaki-Pytorch](https://github.com/lucidrains/phenaki-pytorch), [GR-1](https://github.com/bytedance/GR-1),  [GR1-Training](https://github.com/EDiRobotics/GR1-Training). Thanks for their wonderful works!