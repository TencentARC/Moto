import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import os
from time import time
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import torch
from common.data.datasets import DataPrefetcher
from moto_gpt.src.trainers.trainer_utils import cross_entropy, masked_loss, visualize_latent_motion_gen
import omegaconf
from glob import glob
import shutil
from collections import defaultdict

class MotoGPT_Trainer:
    def __init__(
        self,
        moto_gpt,
        moto_gpt_config,
        latent_motion_tokenizer,
        rgb_preprocessor,
        lang_tokenizer,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.0001,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
        pred_binary_gripper_action=True,
        seed=None
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.accelerator = accelerator
        if seed is not None:
            from transformers import set_seed
            print(f"set_seed: {seed+self.accelerator.process_index}")
            set_seed(seed+self.accelerator.process_index)

        if resume_ckpt_path is not None:
            self.print(f"resuming Moto-GPT from {resume_ckpt_path} ...")

            current_model_dict = moto_gpt.state_dict()
            resume_model_dict = torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu')

            mismatched_param_names = []
            filtered_state_dict = {}

            for name, param in resume_model_dict.items():
                if name in current_model_dict and current_model_dict[name].shape != param.shape:
                    mismatched_param_names.append(name)
                else:
                    filtered_state_dict[name] = param

            missing_keys, unexpected_keys = moto_gpt.load_state_dict(filtered_state_dict, strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            self.print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys, '\nmismatched ', mismatched_param_names)
        optimizer = torch.optim.AdamW(moto_gpt.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes)),
            num_training_steps=num_epochs*total_prints_per_epoch,
        )
        moto_gpt, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            moto_gpt, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        if latent_motion_tokenizer is not None:
            latent_motion_tokenizer = latent_motion_tokenizer.to(accelerator.device)
            latent_motion_tokenizer.eval()
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.moto_gpt = moto_gpt
        self.moto_gpt_config = moto_gpt_config
        self.latent_motion_tokenizer = latent_motion_tokenizer
        self.optimizer = optimizer
        self.train_prefetcher = DataPrefetcher(train_dataloader, self.device, lang_tokenizer=lang_tokenizer)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device, lang_tokenizer=lang_tokenizer)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.lang_tokenizer = lang_tokenizer
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        self.pred_binary_gripper_action = pred_binary_gripper_action


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def save_checkpoint(self, save_dir):
        unwrapped_moto_gpt = self.accelerator.unwrap_model(self.moto_gpt)
        state_dict = unwrapped_moto_gpt.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_moto_gpt.config, os.path.join(save_dir, "config.yaml"))
        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        step = 0
        
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)

                if self.moto_gpt_config.latent_motion_pred:
                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_motion_gen(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {
                'action_arm': torch.tensor(0).float().to(self.device),
                'action_gripper': torch.tensor(0).float().to(self.device),
                'latent_motion': torch.tensor(0).float().to(self.device),
            }
            eval_log_loss = {
                'action_arm': torch.tensor(0).float().to(self.device),
                'action_gripper': torch.tensor(0).float().to(self.device),
                'latent_motion': torch.tensor(0).float().to(self.device),
            }
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            
            while batch is not None:
                with self.accelerator.accumulate(self.moto_gpt):

                    self.moto_gpt.train()
                    self.optimizer.zero_grad()
                    loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['total_loss'])
                    self.optimizer.step()

                    for key in log_loss:
                        log_loss[key] += loss[key].detach() / self.print_steps
                    cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        self.moto_gpt.eval()
                        batch, _ = self.eval_prefetcher.next_without_none()
                        loss = self.calculate_loss(batch, train=True)
                        for key in eval_log_loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    for key in log_loss:
                        log_loss[key] = torch.tensor(0).float().to(self.device)
                    for key in eval_log_loss:
                        eval_log_loss[key] = torch.tensor(0).float().to(self.device)

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()

                if batch_idx  % self.save_steps == 0: #(batch_idx+1)  % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')

                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    if self.moto_gpt_config.latent_motion_pred:
                        visualization_dir = os.path.join(save_dir, 'visualization')
                        self.eval_latent_motion_gen(visualization_dir)
        

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_motion_gen(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        self.moto_gpt.eval()
        batch, _ = self.eval_prefetcher.next_without_none()

        orig_rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
        rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)
        rgb_initial = rgb_seq[:,:1]

        # b, t, c, h, w = batch['rgb_future'].shape
        b, t, c, h, w = rgb_seq.shape
        t = t - 1
        gt_latent_motion_ids = self.latent_motion_tokenizer(
            cond_pixel_values=rgb_seq[:,:-1].reshape(-1, c, h, w),
            target_pixel_values=rgb_seq[:,1:].reshape(-1, c, h, w),
            return_motion_token_ids_only=True
        )
        recons_rgb_future = self.latent_motion_tokenizer.decode_image(
            cond_pixel_values=rgb_seq[:,:-1].reshape(-1, c, h, w),
            given_motion_token_ids=gt_latent_motion_ids
        )["recons_pixel_values"]

        gt_latent_motion_ids = gt_latent_motion_ids.reshape(b, t, -1)
        recons_rgb_future = recons_rgb_future.reshape(b, t, c, h, w)
        recons_rgb_future = self.rgb_preprocessor.post_process(recons_rgb_future)

        decoding_mode2preds = {
            "ground_truth_recons": {
                "frame_preds": recons_rgb_future.detach().cpu(),
                "latent_motion_id_preds": gt_latent_motion_ids.detach().cpu()
            }
        }
        
        decoding_mode2latent_motion_decoding_kwargs = {
            "sampleFalse_beam1": {
                "temperature": 1.0, 
                "sample": False, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 1, 
                "parallel": False
            },

            "sampleTrue_beam5": {
                "temperature": 1.0, 
                "sample": True, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 5, 
                "parallel": False
            },

            "sampleFalse_beam5": {
                "temperature": 1.0, 
                "sample": False, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 5, 
                "parallel": False
            },
        }

        gen_iter_num = 2
        attention_mask = torch.ones(b, t).long().to(self.device)
        dummy_latent_motion_ids = torch.zeros((b, t, gt_latent_motion_ids.shape[-1])).long().to(self.device)
        latent_mask = attention_mask

        for decoding_mode, latent_motion_decoding_kwargs in decoding_mode2latent_motion_decoding_kwargs.items():
            frame_preds = []
            latent_motion_id_preds = []

            cur_cond_pixel_values = rgb_initial.squeeze(1) # (b, c, h, w)
            cur_latent_motion_ids = dummy_latent_motion_ids.clone() # (b, t, per_latent_motion_len)

            cur_rgb_initial = rgb_initial # (b, 1, c, h, w)

            for _ in range(gen_iter_num):
                for buffer_len in range(1, t+1):
                    pred = self.moto_gpt(
                        rgb=cur_rgb_initial, 
                        language=batch['lang_input_ids'],
                        attention_mask=attention_mask,
                        latent_motion_ids=cur_latent_motion_ids,
                        latent_mask=latent_mask,
                        train=False,
                        lang_attention_mask=batch['lang_attention_mask'],
                        buffer_len=buffer_len,
                        **latent_motion_decoding_kwargs,
                    )
                    cur_latent_motion_id_preds = pred['latent_motion_id_preds'] # (b, per_latent_motion_len)
                    cur_latent_motion_ids[:,buffer_len-1] = cur_latent_motion_id_preds
                    cur_frame_preds = self.latent_motion_tokenizer.decode_image(
                        cond_pixel_values=cur_cond_pixel_values,
                        given_motion_token_ids=cur_latent_motion_id_preds
                    )["recons_pixel_values"] # (b, c, h, w)
                    cur_cond_pixel_values = cur_frame_preds
                    frame_preds.append(cur_frame_preds.unsqueeze(1))
                    latent_motion_id_preds.append(cur_latent_motion_id_preds.unsqueeze(1))

                cur_rgb_initial = cur_frame_preds.unsqueeze(1) # (b, 1, c, h, w)

            frame_preds = torch.cat(frame_preds, dim=1)
            frame_preds = self.rgb_preprocessor.post_process(frame_preds)

            latent_motion_id_preds = torch.cat(latent_motion_id_preds, dim=1)
            decoding_mode2preds[decoding_mode] = {
                "frame_preds": frame_preds.detach().cpu(),
                "latent_motion_id_preds": latent_motion_id_preds.detach().cpu()
            }

        orig_rgb_seq = self.rgb_preprocessor.post_process(rgb_seq)
        for i in range(b):
            ith_decoding_mode2preds = defaultdict(dict)
            for decoding_mode, preds in decoding_mode2preds.items():
                for k, v in preds.items():
                    ith_decoding_mode2preds[decoding_mode][k] = v[i]

            visualize_latent_motion_gen(
                lang_goal=batch['lang'][i],
                orig_video=orig_rgb_seq[i], 
                decoding_mode2preds=ith_decoding_mode2preds,
                path=os.path.join(visualization_dir, f"{self.process_index}-{i}")
            )

    def calculate_loss(self, batch, train):
        # image preprocessing
        if self.moto_gpt_config.latent_motion_pred:
            rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
            rgb_seq = self.rgb_preprocessor(rgb_seq, train=train)
            rgb_initial = rgb_seq[:,:1]
        else:
            rgb_initial = self.rgb_preprocessor(batch['rgb_initial'], train=train)

        # obtain ground-truth latent motion ids
        if self.moto_gpt_config.latent_motion_pred:
            # b, t, c, h, w = batch['rgb_future'].shape
            b, t, c, h, w = rgb_seq.shape
            t = t - 1
            latent_motion_ids = self.latent_motion_tokenizer(
                cond_pixel_values=rgb_seq[:,:-1].reshape(-1, c, h, w),
                target_pixel_values=rgb_seq[:,1:].reshape(-1, c, h, w),
                return_motion_token_ids_only=True
            ).reshape(b, t, -1)
        else:
            latent_motion_ids = None

        # compute loss
        attention_mask = batch['mask'][..., 0]
        pred = self.moto_gpt(
            rgb=rgb_initial, # (b, 1, c, h, w)
            language=batch['lang_input_ids'],
            attention_mask=attention_mask, # (b, t)
            latent_motion_ids=latent_motion_ids, # (b, t, per_latent_motion_len)
            latent_mask=batch['latent_mask'], # (b, t)
            train=True,
            lang_attention_mask=batch['lang_attention_mask'],
        )
    
        loss = {}
        device = batch['rgb_initial'].device
        
        if self.moto_gpt_config.get('pred_discrete_arm_action', False): # NOTE: calculate crosee_entropy loss for discrete arm action
            action_arm_loss_func = cross_entropy
            gt_action_arm = batch['actions'][..., :6].long()
        else:
            action_arm_loss_func = F.smooth_l1_loss
            gt_action_arm = batch['actions'][..., :6]
        
        loss['action_arm'] = masked_loss(pred['arm_action_preds'], gt_action_arm, batch['mask'], 0, action_arm_loss_func) if pred['arm_action_preds'] is not None else torch.tensor(0.0).to(device)
        
        if self.pred_binary_gripper_action:
            gripper_action_loss_func = F.binary_cross_entropy_with_logits
        else:
            gripper_action_loss_func = F.smooth_l1_loss
        
        loss['action_gripper'] = masked_loss(pred['gripper_action_preds'], batch['actions'][..., -1:].float(), batch['mask'], 0, gripper_action_loss_func) if pred['gripper_action_preds'] is not None else torch.tensor(0.0).to(device)
        loss['latent_motion'] = masked_loss(pred['latent_motion_preds'], latent_motion_ids, batch['latent_mask'], 0, cross_entropy) if pred['latent_motion_preds'] is not None else torch.tensor(0.0).to(device)
        total_loss = 100 * loss['action_arm'] + loss['action_gripper'] + loss['latent_motion']
        loss['total_loss'] = total_loss
        return loss


    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*(self.moto_gpt_config.sequence_length+1)) / (time()-clock)
        fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}_loss: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}_loss: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key+'_loss', log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key+'_loss', eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)
