import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from moto_gpt.src.trainers.moto_gpt_trainer import MotoGPT_Trainer
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset

def main(cfg):
    # Prepare Moto-GPT
    seed = cfg['training_config'].get('seed', None)
    if seed is not None:
        from transformers import set_seed
        print(f"set_seed: {seed}")
        set_seed(seed)

    moto_gpt_config_path = cfg['moto_gpt_config_path']
    print(f"initializing Moto-GPT from {moto_gpt_config_path} ...")
    moto_gpt_config = omegaconf.OmegaConf.load(moto_gpt_config_path)
    moto_gpt = hydra.utils.instantiate(moto_gpt_config)
    moto_gpt.config = moto_gpt_config

    # Prepare lang_tokenizer and rgb_processor
    lang_tokenizer = AutoTokenizer.from_pretrained(moto_gpt_config['model_lang']['pretrained_model_name_or_path'])
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Prepare Latent Motion Tokenizer
    if moto_gpt_config['latent_motion_pred']:
        latent_motion_tokenizer_path = cfg['latent_motion_tokenizer_path']
        assert latent_motion_tokenizer_path is not None
        print(f"loading Latent Motion Tokenizer from {latent_motion_tokenizer_path} ...")
        latent_motion_tokenizer = load_model(latent_motion_tokenizer_path)
    else:
        latent_motion_tokenizer=None

    vq_remap = cfg.get('vq_remap', None)
    unknown_index = cfg.get('unknown_index', 'closest')
    if vq_remap is not None:
        latent_motion_tokenizer.vector_quantizer.setup_remap(remap=vq_remap, unknown_index=unknown_index)

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    
    extra_data_config = {
        'sequence_length': moto_gpt_config['sequence_length'],
        'chunk_size': moto_gpt_config['chunk_size'],
        'act_dim': moto_gpt_config['act_dim'],
        'do_extract_future_frames': moto_gpt_config['latent_motion_pred'],
        'do_extract_action': moto_gpt_config['act_pred']
    }

    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    # Prepare Trainer
    trainer = MotoGPT_Trainer(
        moto_gpt=moto_gpt,
        moto_gpt_config=moto_gpt_config,
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        lang_tokenizer=lang_tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/group/40101/milkcychen/Moto/moto_gpt/configs/train/data_calvin-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0002_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch10.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    


