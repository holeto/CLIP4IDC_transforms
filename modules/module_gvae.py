"""Grammar VAE (Kusner et al., 2017), ported from the Keras code in ../Grammar_VAE.

A datapoint is the leftmost derivation of a parse tree: a sequence of T rule
indices, one-hot over the K rules and padded with a no-op rule. The encoder is
a Conv1d stack over the one-hot, the decoder a GRU stack that outputs logits
for all T steps at once from z (not autoregressive). The grammar enters only
through masks: at every step the softmax is restricted to the rules allowed
there, both in the loss (masks of the true derivation) and in decoding (masks
of the derivation decoded so far).

Nothing here knows the grammar: masks come in as (B, T, K) bool tensors, and
decoding asks per-sample masker objects with mask(), apply(rule) and done
(modules/trance_grammar.py has the TRANCE ones).

Differences from the Keras code (models/model_eq.py):
- The reconstruction loss is the categorical cross-entropy of the true rules
  under the masked softmax, summed over T (the paper's objective). Keras
  applies binary cross-entropy to every entry of the masked distribution.
- Decoding is greedy by default; Gumbel sampling, as in _sample_using_masks,
  is the sample option.
"""
import torch
from torch import nn
import torch.nn.functional as F

DEFAULT_HYPERS = {
    "conv": ((32, 2), (32, 3), (32, 4)),
    "dense": 256,
    "hidden": 256,
}


def _keras_batch_norm(n):
    # Keras defaults: moving average 0.99, epsilon 1e-3.
    return nn.BatchNorm1d(n, eps=1e-3, momentum=0.01)


class GVAEEncoder(nn.Module):
    """One-hot derivation (B, T, K) -> (z_mean, z_log_var), each (B, latent_dim)."""

    def __init__(self, n_rules, max_len, latent_dim, hypers):
        super(GVAEEncoder, self).__init__()
        layers = []
        channels, length = n_rules, max_len
        for filters, kernel in hypers["conv"]:
            layers += [nn.Conv1d(channels, filters, kernel), nn.ReLU(), _keras_batch_norm(filters)]
            channels, length = filters, length - kernel + 1
        self.conv = nn.Sequential(*layers)
        self.dense = nn.Linear(channels * length, hypers["dense"])
        self.z_mean = nn.Linear(hypers["dense"], latent_dim)
        self.z_log_var = nn.Linear(hypers["dense"], latent_dim)

    def forward(self, x):
        # Keras convolves along time with the rules as channels.
        h = self.conv(x.transpose(1, 2))
        h = F.relu(self.dense(h.flatten(1)))
        return self.z_mean(h), self.z_log_var(h)


class GVAEDecoder(nn.Module):
    """z (B, latent_dim) -> unnormalised rule logits (B, T, K)."""

    def __init__(self, n_rules, max_len, latent_dim, hypers):
        super(GVAEDecoder, self).__init__()
        self.max_len = max_len
        self.norm = _keras_batch_norm(latent_dim)
        self.latent_input = nn.Linear(latent_dim, latent_dim)
        self.gru = nn.GRU(latent_dim, hypers["hidden"], num_layers=3, batch_first=True)
        self.out = nn.Linear(hypers["hidden"], n_rules)

    def forward(self, z):
        h = F.relu(self.latent_input(self.norm(z)))
        # Keras RepeatVector: the same input at every time step.
        h, _ = self.gru(h.unsqueeze(1).expand(-1, self.max_len, -1).contiguous())
        return self.out(h)


class GrammarVAE(nn.Module):
    def __init__(self, n_rules, max_len, latent_dim=56, hypers=None, epsilon_std=0.01):
        """n_rules: K, the rules of the grammar including the no-op. max_len: T.

        epsilon_std scales the noise of the reparameterisation; the Keras code
        uses 0.01, so the decoder sees z close to z_mean.
        """
        super(GrammarVAE, self).__init__()
        hypers = dict(DEFAULT_HYPERS, **(hypers or {}))
        self.n_rules = n_rules
        self.max_len = max_len
        self.epsilon_std = epsilon_std
        self.encoder = GVAEEncoder(n_rules, max_len, latent_dim, hypers)
        self.decoder = GVAEDecoder(n_rules, max_len, latent_dim, hypers)
        self.apply(_keras_init)

    def one_hot(self, rules):
        return F.one_hot(rules, self.n_rules).float()

    def encode(self, rules):
        """Rule indices (B, T) -> (z_mean, z_log_var)."""
        return self.encoder(self.one_hot(rules))

    def reparameterize(self, z_mean, z_log_var):
        if not self.training:
            return z_mean
        eps = torch.randn_like(z_mean) * self.epsilon_std
        return z_mean + torch.exp(z_log_var / 2) * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, rules):
        """Returns (logits, z_mean, z_log_var)."""
        z_mean, z_log_var = self.encode(rules)
        return self.decode(self.reparameterize(z_mean, z_log_var)), z_mean, z_log_var


def _keras_init(module):
    # Keras 1 defaults: glorot uniform kernels, orthogonal recurrent kernels,
    # zero biases.
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if name.startswith("weight_ih"):
                for gate in param.chunk(3, 0):
                    nn.init.xavier_uniform_(gate)
            elif name.startswith("weight_hh"):
                for gate in param.chunk(3, 0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(param)


def masked_log_softmax(logits, mask):
    """Log-softmax restricted to the allowed rules; -inf elsewhere.

    The Keras code multiplies exp(logits) by the mask and renormalises, which
    is the same distribution. Every row of mask must allow at least one rule.
    """
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def reconstruction_loss(logits, rules, mask):
    """Negative log-likelihood of the derivations, summed over T, mean over B.

    mask must allow the true rule at every step (teacher masks), so the loss is
    finite. Padding steps allow only the no-op and add exactly 0.
    """
    log_p = masked_log_softmax(logits, mask).gather(-1, rules.unsqueeze(-1)).squeeze(-1)
    return -log_p.sum(1).mean()


def kl_loss(z_mean, z_log_var):
    """KL to N(0, I), averaged over the latent dimensions as in the Keras code."""
    kl = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
    return kl.mean()


def masked_greedy_decode(logits, maskers, sample=False):
    """Decode rule logits (B, T, K) under per-sample maskers.

    At every step each masker gives the allowed rules for the derivation so
    far; the best allowed rule (or a Gumbel sample, if sample) is applied.
    Returns the T rules of every sample. A sample whose masker is not done
    afterwards ran out of steps; one that allows no rule stops early, its
    remaining steps are -1.
    """
    batch, max_len, _ = logits.shape
    logits = logits.detach().float().cpu()
    if sample:
        logits = logits - torch.empty_like(logits).exponential_().log()
    rules = torch.full((batch, max_len), -1, dtype=torch.long)
    alive = [True] * batch
    for t in range(max_len):
        for b, masker in enumerate(maskers):
            if not alive[b]:
                continue
            mask = torch.tensor(masker.mask(), dtype=torch.bool)
            if not mask.any():
                alive[b] = False
                continue
            rule = int(logits[b, t].masked_fill(~mask, float("-inf")).argmax())
            masker.apply(rule)
            rules[b, t] = rule
    return rules
