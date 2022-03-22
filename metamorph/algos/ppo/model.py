import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from metamorph.config import cfg
from metamorph.utils import model as tu

from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS
        # Embedding layer for per limb obs
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        self.limb_embed = nn.Linear(limb_obs_size, self.d_model)
        self.ext_feat_fusion = self.model_args.EXT_MIX

        if self.model_args.POS_EMBEDDING == "learnt":
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.model_args.POS_EMBEDDING == "abs":
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim

        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = tu.make_mlp_default(
            [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
            final_nonlinearity=False,
        )
        self.init_weights()

    def init_weights(self):
        initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, return_attention=False):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        obs_embed = self.limb_embed(obs) * math.sqrt(self.d_model)
        _, batch_size, _ = obs_embed.shape

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

        attention_maps = None

        if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, src_key_padding_mask=obs_mask
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, src_key_padding_mask=obs_mask
            )

        decoder_input = obs_embed_t
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        output = self.decoder(decoder_input)
        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS
        self.v_net = TransformerModel(obs_space, 1)

        if cfg.ENV_NAME == "Unimal-v0":
            self.mu_net = TransformerModel(obs_space, 2)
            self.num_actions = cfg.MODEL.MAX_LIMBS * 2
        else:
            raise ValueError("Unsupported ENV_NAME")

        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

    def forward(self, obs, act=None, return_attention=False):
        if act is not None:
            batch_size = cfg.PPO.BATCH_SIZE
        else:
            batch_size = cfg.PPO.NUM_ENVS

        obs_env = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        if "obs_padding_cm_mask" in obs:
            obs_cm_mask = obs["obs_padding_cm_mask"]
        else:
            obs_cm_mask = None
        obs, obs_mask, act_mask, _ = (
            obs["proprioceptive"],
            obs["obs_padding_mask"],
            obs["act_padding_mask"],
            obs["edges"],
        )

        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()

        obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        # Per limb critic values
        limb_vals, v_attention_maps = self.v_net(
            obs, obs_mask, obs_env, obs_cm_mask, return_attention=return_attention
        )
        # Zero out mask values
        limb_vals = limb_vals * (1 - obs_mask.int())
        # Use avg/max to keep the magnitidue same instead of sum
        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)

        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, return_attention=return_attention
        )
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, None, None


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs):
        val, pi, _, _ = self.ac(obs)
        act = pi.sample()
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, _, _, _ = self.ac(obs)
        return val
