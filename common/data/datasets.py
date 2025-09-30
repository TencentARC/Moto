import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import decode_jpeg
import cv2
import os
from einops import rearrange
import random
import json
from PIL import Image
import random

def get_split_and_ratio(split, splits):
    assert split in ['train', 'val']
    assert 'train' in splits
    if 'val' in splits:
        start_ratio=0
        end_ratio=1
    else:
        if split == 'train':
            start_ratio=0
            end_ratio=0.95
        else:
            split = 'train' 
            start_ratio=0.95
            end_ratio=1
    return split, start_ratio, end_ratio

class DataPrefetcher:
    def __init__(self, loader, device, lang_tokenizer=None):
        self.loader = loader
        self.device = device
        self.lang_tokenizer = lang_tokenizer
        self.iter = iter(self.loader)

    def __len__(self):
        return len(self.loader.dataset)

    def next(self):
        clock = time()
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return None, time()-clock

        if batch is not None:
            if self.lang_tokenizer is not None:
                lang_inputs = self.lang_tokenizer(batch['lang'], return_tensors="pt", padding=True)
                lang_input_ids = lang_inputs.input_ids
                lang_attention_mask = lang_inputs.attention_mask
                batch["lang_input_ids"] = lang_input_ids
                batch["lang_attention_mask"] = lang_attention_mask

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time


class LMDBDataset_Mix(Dataset):
    def __init__(self, datasets, sample_weights):
        super().__init__()
        self.datasets = datasets
        self.sample_weights = np.array(sample_weights)
        self.num_datasets = len(datasets)
        self.dataset_sizes = []
        for dataset in self.datasets:
            self.dataset_sizes.append(len(dataset))

    def __getitem__(self, idx):
        dataset_index = np.random.choice(self.num_datasets, p=self.sample_weights / self.sample_weights.sum())
        # idx is not used
        idx = np.random.randint(self.dataset_sizes[dataset_index])
        return self.datasets[dataset_index][idx]

    def __len__(self):
        return sum(self.dataset_sizes)


class LMDBDataset_for_MotoGPT(Dataset):
    def __init__(
        self, lmdb_dir, split, skip_frame, 
        sequence_length, #start_ratio, end_ratio, 
        chunk_size=3, act_dim=7, 
        do_extract_future_frames=True, do_extract_action=False,
        video_dir=None, rgb_shape=(224, 224), rgb_preprocessor=None, max_skip_frame=None,
        no_repeat_data=False):


        super().__init__()

        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.do_extract_future_frames = do_extract_future_frames
        self.do_extract_action = do_extract_action

        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, act_dim)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        self.lmdb_dir = lmdb_dir
        self.video_dir = video_dir
        self.rgb_preprocessor = rgb_preprocessor
        self.no_repeat_data = no_repeat_data

        split, start_ratio, end_ratio = get_split_and_ratio(split, os.listdir(lmdb_dir))
        self.split = split
        env = lmdb.open(os.path.join(lmdb_dir, split), readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length * skip_frame - chunk_size
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(os.path.join(self.lmdb_dir, self.split), readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def extract_lang_goal(self, idx, cur_episode):
        feature_dict = loads(self.txn.get(f'feature_dict_{idx}'.encode()))
        lang = feature_dict['observation']['natural_language_instruction'].decode().lower().strip('.')
        return lang

    def get_video_path(self, cur_episode):
        # return os.path.join(self.video_dir, f'{self.split}_eps_{cur_episode:08d}.mp4')
        raise NotImplementedError

    def extract_frames(self, idx, cur_episode, delta_t, rgb_initial, rgb_future, latent_mask):
        start_local_step = loads(self.txn.get(f'local_step_{idx}'.encode()))
        video_path = self.get_video_path(cur_episode)
        video = cv2.VideoCapture(video_path)

        def _extract_frame(frame_idx):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            try:
                assert ret is True
            except Exception as e:
                # print(f"Failed to read video (path={video_path}, frame_idx={frame_idx})")
                raise e
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        rgb_initial[0] = _extract_frame(start_local_step)

        if self.do_extract_future_frames:
            for i in range(self.sequence_length):
                if loads(self.txn.get(f'cur_episode_{idx+(i+1)*delta_t}'.encode())) == cur_episode:
                    rgb_future[i] = _extract_frame(start_local_step+(i+1)*delta_t)
                    latent_mask[i] = 1
                else:
                    break

        video.release()

    def extract_actions(self, idx, cur_episode, delta_t, actions, mask):
        for i in range(self.sequence_length):
            for j in range(self.chunk_size):
                cur_idx = idx + i*delta_t + j
                if loads(self.txn.get(f'cur_episode_{cur_idx}'.encode())) == cur_episode:
                    mask[i, j] = 1
                    action = self.extract_action(cur_idx)
                    actions[i, j] = action

    def extract_action(self, idx):
        raise NotImplementedError

    
    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        while True:
            try:
                orig_idx = idx
                idx = idx + self.start_step
                cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))

                if self.max_skip_frame is None:
                    delta_t = self.skip_frame
                else:
                    delta_t = random.randint(self.skip_frame, self.max_skip_frame)

                # dummy features
                rgb_initial = self.dummy_rgb_initial.clone()
                rgb_future = self.dummy_rgb_future.clone()
                actions = self.dummy_actions.clone()
                mask = self.dummy_mask.clone()
                latent_mask = self.dummy_latent_mask.clone()

                # extract lang goal
                lang = self.extract_lang_goal(idx, cur_episode)

                # extract initial frame and future frames
                self.extract_frames(
                    idx=idx, cur_episode=cur_episode, delta_t=delta_t,
                    rgb_initial=rgb_initial, 
                    rgb_future=rgb_future, latent_mask=latent_mask
                )

                # extract actions
                if self.do_extract_action:
                    self.extract_actions(
                        idx=idx, cur_episode=cur_episode, delta_t=delta_t,
                        actions=actions, 
                        mask=mask
                    )

                if not self.no_repeat_data:
                    if self.do_extract_future_frames and (not self.do_extract_action) and latent_mask.sum() == 0:
                        raise Exception("latent_mask should be larger than zero!")

                return {
                    "lang": lang,
                    "rgb_initial": rgb_initial,
                    "rgb_future": rgb_future,
                    "actions": actions,
                    "mask": mask,
                    "latent_mask": latent_mask,
                    "idx": orig_idx
                }
                    
            except Exception as e:
                # print(e)
                idx = random.randint(0, len(self))
                # print(idx)
            

    def __len__(self):
        return self.end_step - self.start_step

class LMDBDataset_for_MotoGPT_OXE(LMDBDataset_for_MotoGPT):
    def get_video_path(self, cur_episode):
        return os.path.join(self.video_dir, f'{self.split}_eps_{cur_episode:08d}.mp4')


class LMDBDataset_for_MotoGPT_Video(LMDBDataset_for_MotoGPT):
    def get_video_path(self, cur_episode):
        return os.path.join(self.video_dir, cur_episode)


class LMDBDataset_for_MotoGPT_RT1(LMDBDataset_for_MotoGPT_OXE):
    def __init__(self, world_vector_range=(-1.0, 1.0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_vector_range = world_vector_range

    def extract_action(self, idx):
        feature_dict = loads(self.txn.get(f'feature_dict_{idx}'.encode()))
        action = []
        for act_name, act_min, act_max in [
            ('world_vector', self.world_vector_range[0], self.world_vector_range[1]),
            ('rotation_delta', -np.pi / 2, np.pi / 2),
            ('gripper_closedness_action', -1.0, 1.0)
        ]:
            action.append(np.clip(feature_dict['action'][act_name], act_min, act_max))
        action = np.concatenate(action)
        action = torch.from_numpy(action)
        return action


class LMDBDataset_for_MotoGPT_CALVIN(LMDBDataset_for_MotoGPT):
    def extract_lang_goal(self, idx, cur_episode):
        lang = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        return lang

    def extract_frames(self, idx, cur_episode, delta_t, rgb_initial, rgb_future, latent_mask):
        rgb_initial[0] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx}'.encode())))

        if self.do_extract_future_frames:
            for i in range(self.sequence_length):
                if loads(self.txn.get(f'cur_episode_{idx+(i+1)*delta_t}'.encode())) == cur_episode:
                    rgb_future[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx+(i+1)*delta_t}'.encode())))
                    latent_mask[i] = 1
                else:
                    break

    def extract_actions(self, idx, cur_episode, delta_t, actions, mask):
        for i in range(self.sequence_length):
            for j in range(self.chunk_size):
                cur_idx = idx + i*delta_t + j
                if loads(self.txn.get(f'cur_episode_{cur_idx}'.encode())) == cur_episode:
                    mask[i, j] = 1
                    action = self.extract_action(cur_idx)
                    actions[i, j] = action

    def extract_action(self, idx):
        action = loads(self.txn.get(f'rel_action_{idx}'.encode()))
        action[-1] = (action[-1] + 1) / 2
        return action









class JsonDataset_for_MotoGPT_Video(Dataset):
    def __init__(
        self, split, skip_frame, 
        sequence_length, #start_ratio, end_ratio,
        video_dir=None, rgb_shape=(224, 224), 
        rgb_preprocessor=None, max_skip_frame=None, video_metadata_path=None, *args, **kwargs):

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame

        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        self.video_dir = video_dir
        self.rgb_preprocessor = rgb_preprocessor

        if video_metadata_path is None:
            video_metadata_path = os.path.join(video_dir, 'video_metadata.json')
        else:
            print(f"specified video_metadata_path: {video_metadata_path}")
        
        with open(video_metadata_path) as f:
            video_metadata = json.load(f)

        split, start_ratio, end_ratio = get_split_and_ratio(split, video_metadata.keys())
        self.split = split

        video_metadata = video_metadata[split]
        videos = video_metadata['videos']
        start_step = int(len(videos) * start_ratio) 
        end_step = int(len(videos) * end_ratio)
        self.videos = videos[start_step:end_step]
        self.num_videos = len(self.videos)
        total_frames = video_metadata['total_frames']
        self.dataset_len = int(total_frames*(end_ratio-start_ratio)) - skip_frame * self.num_videos

    def get_video_path(self, video_basename):
        return os.path.join(self.video_dir, video_basename)

    def extract_frames(self, video_basename, start_local_step, num_frames, delta_t, 
                       rgb_initial, rgb_future, latent_mask):
        video_path = self.get_video_path(video_basename)
        video = cv2.VideoCapture(video_path)

        def _extract_frame(frame_idx):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            try:
                assert ret is True
            except Exception as e:
                raise e
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        rgb_initial[0] = _extract_frame(start_local_step)

        for i in range(self.sequence_length):
            next_local_step = start_local_step+(i+1)*delta_t
            if next_local_step < num_frames:
                rgb_future[i] = _extract_frame(next_local_step)
                latent_mask[i] = 1
            else:
                break

        video.release()

    
    def obtain_item(self, idx, start_local_step=None, delta_t=None):
        video_basename, num_frames, *_ = self.videos[idx]

        assert self.skip_frame < num_frames
        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                max_skip_frame = min(num_frames-1, self.max_skip_frame)
                delta_t = random.randint(self.skip_frame, max_skip_frame)

        # dummy features
        rgb_initial = self.dummy_rgb_initial.clone()
        rgb_future = self.dummy_rgb_future.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        if start_local_step is None:
            start_local_step = random.randint(0, num_frames-1-delta_t)

        self.extract_frames(
            video_basename=video_basename, start_local_step=start_local_step, num_frames=num_frames,
            delta_t=delta_t,
            rgb_initial=rgb_initial, 
            rgb_future=rgb_future, latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        return {
            "rgb_initial": rgb_initial,
            "rgb_future": rgb_future,
            "latent_mask": latent_mask,
            "idx": idx,
            "delta_t": delta_t,
            "start_local_step": start_local_step
        }

    
    def __getitem__(self, idx):
        while True:
            try:
                video_idx = idx % self.num_videos
                return self.obtain_item(video_idx)
            except Exception as e:
                # print(e)
                idx = random.randint(0, len(self)-1)
            

    def __len__(self):
        return self.dataset_len








class NpzDataset_for_MotoGPT_Video(Dataset):
    def __init__(
        self, split, skip_frame, 
        sequence_length,
        npz_dir=None, rgb_shape=(224, 224), 
        rgb_preprocessor=None, max_skip_frame=None, npz_metadata_path=None, *args, **kwargs):

        super().__init__()

        self.sequence_length = sequence_length
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_latent_mask = torch.zeros(sequence_length)

        if split == 'train':
            split = 'training'
        elif split == 'val':
            split = 'validation'
        else:
            raise NotImplementedError

        self.npz_dir = os.path.join(npz_dir, split)
        self.rgb_preprocessor = rgb_preprocessor

        if npz_metadata_path is None:
            npz_metadata_path = os.path.join(self.npz_dir, 'npz_metadata.json')
        else:
            print(f"specified npz_metadata_path: {npz_metadata_path}")
        
        with open(npz_metadata_path) as f:
            npz_metadata = json.load(f)

        self.npz_metadata = npz_metadata
        self.dataset_len = len(npz_metadata) - skip_frame

    def get_npz_path(self, npz_basename):
        return os.path.join(self.npz_dir, npz_basename)

    def extract_frames(self, npz_basename, delta_t, 
                       rgb_initial, rgb_future, latent_mask):
        
        def _extract_frame(npz_idx):
            npz_path = self.get_npz_path(f"episode_{str(npz_idx).zfill(7)}.npz")
            try:
                frame = Image.fromarray(np.load(npz_path)['rgb_static']).convert("RGB")
            except Exception as e:
                raise e

            frame = np.array(frame)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        start_npz_path = self.get_npz_path(npz_basename)
        start_npz_idx = int(npz_basename.split("_")[-1].split(".")[0])
        rgb_initial[0] = _extract_frame(start_npz_idx)

        for i in range(self.sequence_length):
            next_npz_idx = start_npz_idx+(i+1)*delta_t
            try:
                rgb_future[i] = _extract_frame(next_npz_idx)
                latent_mask[i] = 1
            except:
                break

    
    def obtain_item(self, idx, delta_t=None):
        npz_basename = self.npz_metadata[idx]
        npz_idx = int(npz_basename.split("_")[-1].split(".")[0])

        if delta_t is None:
            if self.max_skip_frame is None:
                delta_t = self.skip_frame
            else:
                delta_t = random.randint(self.skip_frame, self.max_skip_frame)

        # dummy features
        rgb_initial = self.dummy_rgb_initial.clone()
        rgb_future = self.dummy_rgb_future.clone()
        latent_mask = self.dummy_latent_mask.clone()

        # extract initial frame and future frames
        self.extract_frames(
            npz_basename=npz_basename,
            delta_t=delta_t,
            rgb_initial=rgb_initial, 
            rgb_future=rgb_future, 
            latent_mask=latent_mask
        )

        if latent_mask.sum() == 0:
            raise Exception("latent_mask should be larger than zero!")

        return {
            "rgb_initial": rgb_initial,
            "rgb_future": rgb_future,
            "latent_mask": latent_mask,

            "idx": idx,
            "delta_t": delta_t,
        }

    
    def __getitem__(self, idx):
        while True:
            try:
                return self.obtain_item(idx)
            except Exception as e:
                idx = random.randint(0, len(self)-1)
            

    def __len__(self):
        return self.dataset_len