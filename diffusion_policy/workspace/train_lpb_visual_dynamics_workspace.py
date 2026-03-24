import copy
import os
import random
from collections import OrderedDict

import hydra
import numpy as np
import torch
import tqdm
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainLPBVisualDynamicsWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch", "normalizer"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        _seed_everything(cfg.training.seed)
        self.global_step = 0
        self.epoch = 0

        self.encoder = hydra.utils.instantiate(
            cfg.encoder,
            policy_ckpt_path=cfg.reference_policy.checkpoint,
            view_names=cfg.task.view_names,
        )
        if cfg.model.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            cfg.proprio_encoder,
            num_frames=cfg.num_hist + cfg.num_pred,
            in_chans=cfg.task.proprio_dim,
            emb_dim=cfg.model.proprio_emb_dim,
        )
        self.action_encoder = hydra.utils.instantiate(
            cfg.action_encoder,
            num_frames=cfg.num_hist + cfg.num_pred,
            in_chans=cfg.task.original_action_dim * cfg.frameskip,
            emb_dim=cfg.model.action_emb_dim,
        )
        self.predictor = hydra.utils.instantiate(
            cfg.predictor,
            num_patches=1,
            num_frames=cfg.num_hist,
            dim=self.encoder.emb_dim * len(cfg.task.view_names)
            + cfg.model.proprio_emb_dim
            + cfg.model.action_emb_dim,
            visual_dim=self.encoder.emb_dim * len(cfg.task.view_names),
            proprio_dim=cfg.model.proprio_emb_dim,
            action_dim=cfg.model.action_emb_dim,
        )
        self.model = hydra.utils.instantiate(
            cfg.model.net,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            proprio_dim=cfg.model.proprio_emb_dim,
            action_dim=cfg.model.action_emb_dim,
            view_names=cfg.task.view_names,
        )
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.normalizer = None

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        dataset = hydra.utils.instantiate(cfg.task.dataset, train=True)
        val_dataset = hydra.utils.instantiate(cfg.task.dataset, train=False)
        self.normalizer = dataset.get_normalizer().to(device)
        torch.save(self.normalizer.state_dict(), os.path.join(self.output_dir, "normalizer.pth"))

        train_loader = DataLoader(dataset, **cfg.dataloader)
        val_loader = DataLoader(val_dataset, **cfg.val_dataloader)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                train_log = self._run_epoch(train_loader, device, train=True)
                val_log = self._run_epoch(val_loader, device, train=False)
                step_log = {
                    **train_log,
                    **val_log,
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                }
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    self.save_checkpoint(tag="latest")

                self.global_step += 1
                self.epoch += 1

    def _run_epoch(self, dataloader, device, train=True):
        phase = "train" if train else "val"
        losses = []
        z_losses = []
        z_visual_losses = []
        z_proprio_losses = []
        self.model.train(train)
        if self.cfg.model.freeze_encoder:
            self.encoder.eval()

        iterator = tqdm.tqdm(dataloader, desc=f"{phase} epoch {self.epoch}", leave=False)
        for obs, act, _state in iterator:
            obs = self._prepare_obs(obs, device)
            act = self._prepare_actions(act, device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_components = self.model(obs, act)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss, loss_components = self.model(obs, act)
            losses.append(loss.item())
            z_losses.append(loss_components["z_loss"].item())
            z_visual_losses.append(loss_components["z_visual_loss"].item())
            z_proprio_losses.append(loss_components["z_proprio_loss"].item())

        return {
            f"{phase}_loss": float(np.mean(losses)) if losses else 0.0,
            f"{phase}_z_loss": float(np.mean(z_losses)) if z_losses else 0.0,
            f"{phase}_z_visual_loss": float(np.mean(z_visual_losses)) if z_visual_losses else 0.0,
            f"{phase}_z_proprio_loss": float(np.mean(z_proprio_losses)) if z_proprio_losses else 0.0,
        }

    def _prepare_obs(self, obs, device):
        for view_name in self.cfg.task.view_names:
            obs["visual"][view_name] = obs["visual"][view_name].to(device)
            obs["visual"][view_name] = self.normalizer[view_name].normalize(obs["visual"][view_name])
        obs["proprio"] = self.normalizer["state"].normalize(obs["proprio"].to(device))
        return obs

    def _prepare_actions(self, act, device):
        act = self.normalizer["act"].normalize(act.to(device))
        act = rearrange(
            act,
            "b (n f) d -> b n (f d)",
            n=self.cfg.num_hist + self.cfg.num_pred,
            d=self.cfg.task.original_action_dim,
        )
        act[:, -1:, :] = 0
        return act
