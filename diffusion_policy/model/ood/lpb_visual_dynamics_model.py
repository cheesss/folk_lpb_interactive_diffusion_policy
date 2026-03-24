import torch
import torch.nn as nn
from einops import rearrange, repeat


class LPBVisualDynamicsModel(nn.Module):
    def __init__(
        self,
        image_size,
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        train_encoder=True,
        train_predictor=False,
        view_names=None,
        use_layernorm=True,
        language_encoder=None,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.view_names = view_names or ["view1"]
        self.language_encoder = language_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.emb_dim = self.encoder.emb_dim * len(self.view_names) + (self.action_dim + self.proprio_dim)
        self.emb_criterion = nn.MSELoss()

        if use_layernorm:
            self.per_view_norm = nn.ModuleDict({
                view_name: nn.LayerNorm(self.encoder.emb_dim, elementwise_affine=False)
                for view_name in self.view_names
            })
            if len(self.view_names) > 1:
                total_dim = self.encoder.emb_dim * len(self.view_names)
                self.fusion_norm = nn.LayerNorm(total_dim, elementwise_affine=False)

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()

    def encode(self, obs, act):
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        proprio_tiled = repeat(z_dct["proprio"].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct["visual"].shape[2])
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct["visual"].shape[2])
        if "language" not in z_dct:
            z = torch.cat([z_dct["visual"], proprio_tiled, act_tiled], dim=3)
        else:
            language_repeated = z_dct["language"].unsqueeze(1).repeat(1, proprio_tiled.shape[1], 1, 1)
            z = torch.cat([z_dct["visual"], proprio_tiled, act_tiled, language_repeated], dim=3)
        return z

    def encode_act(self, act):
        return self.action_encoder(act)

    def encode_proprio(self, proprio):
        return self.proprio_encoder(proprio)

    def encode_obs(self, obs):
        visual_embs = self.encode_obs_visual(obs["visual"])
        proprio = obs["proprio"]
        proprio_emb = self.encode_proprio(proprio)
        res = {"visual": visual_embs, "proprio": proprio_emb}
        if "language" in obs:
            res["language"] = obs["language"]
        return res

    def encode_obs_visual(self, obs_visual):
        view_embs = self.encoder(obs_visual)
        for view_name in self.view_names:
            if hasattr(self, "per_view_norm"):
                view_embs[view_name] = self.per_view_norm[view_name](view_embs[view_name])
        visual_embs = torch.cat([view_embs[view_name] for view_name in self.view_names], dim=-1)
        if hasattr(self, "fusion_norm"):
            visual_embs = self.fusion_norm(visual_embs)
        return visual_embs

    def predict(self, z):
        t = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=t)
        return z

    def separate_emb(self, z):
        z_visual = z[..., : -(self.proprio_dim + self.action_dim)]
        z_proprio = z[..., -(self.proprio_dim + self.action_dim) : -self.action_dim]
        z_act = z[..., -self.action_dim :]
        z_proprio = z_proprio[:, :, 0, : self.proprio_dim]
        z_act = z_act[:, :, 0, : self.action_dim]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]
        z_tgt = z[:, self.num_pred :, :, :]
        if "language" in obs:
            z_tgt = z_tgt[..., :-32]

        z_pred = self.predict(z_src)
        z_visual_loss = self.emb_criterion(
            z_pred[:, :, :, : -(self.proprio_dim + self.action_dim)],
            z_tgt[:, :, :, : -(self.proprio_dim + self.action_dim)].detach(),
        )
        z_proprio_loss = self.emb_criterion(
            z_pred[:, :, :, -(self.proprio_dim + self.action_dim) : -self.action_dim],
            z_tgt[:, :, :, -(self.proprio_dim + self.action_dim) : -self.action_dim].detach(),
        )
        z_loss = self.emb_criterion(
            z_pred[:, :, :, : -self.action_dim],
            z_tgt[:, :, :, : -self.action_dim].detach(),
        )
        loss = z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss
        loss_components["loss"] = loss
        return loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
        return torch.cat([z[..., : -self.action_dim], act_tiled], dim=-1)
