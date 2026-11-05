"""CLIP4IDC with the grammar target (--target grammar, TRANCE only).

The transformation is a derivation of the TRANCE grammar instead of a text,
and a Grammar VAE (module_gvae.py) takes the place of CLIP's text tower. The
image-pair embedding is projected into the GVAE latent space (--latent_dim, by
CLIP4IDC.visual_to_latent, shared with the text target):
- stage 1 (retrieval): symmetric InfoNCE between the projected pair embedding
  and the GVAE z_mean of the derivation, plus the GVAE loss (masked
  reconstruction and KL) of the same derivations;
- stage 2 (caption): the projected pair embedding is the z of the GVAE
  decoder, trained with the masked reconstruction loss and decoded under the
  mask; --freeze_gvae_decoder keeps the decoder of stage 1.
CLIP's text tower is still built and loaded, but unused.
"""
import torch

from modules.modeling import CLIP4IDC
from modules.module_gvae import GrammarVAE, kl_loss, masked_greedy_decode, reconstruction_loss
from modules import trance_grammar


class CLIP4IDCGrammar(CLIP4IDC):
    def __init__(self, cross_config, decoder_config, clip_state_dict, task_config):
        super(CLIP4IDCGrammar, self).__init__(cross_config, decoder_config, clip_state_dict, task_config)
        assert self.latent_dim > 0, "the grammar target needs --latent_dim > 0"
        self.gvae = GrammarVAE(trance_grammar.K, trance_grammar.MAX_LEN, self.latent_dim,
                               hypers={"hidden": task_config.gvae_hidden}, epsilon_std=task_config.gvae_epsilon_std)
        # Detached terms of the last training loss, for logging.
        self.loss_terms = {}
        # Stage 2 option: the decoder stays the stage-1 one, see train().
        self.freeze_gvae_decoder = self._stage_two and getattr(task_config, "freeze_gvae_decoder", False)
        if self.freeze_gvae_decoder:
            for param in self.gvae.decoder.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(CLIP4IDCGrammar, self).train(mode)
        # A frozen decoder also keeps its BatchNorm statistics, which training
        # mode would replace by those of the image embeddings. Only the
        # BatchNorm: cuDNN runs the GRU's backward only in training mode, and
        # the GRU has no dropout, so its modes compute the same.
        if self.freeze_gvae_decoder:
            self.gvae.decoder.norm.eval()
        return self

    def forward(self, input_ids, token_type_ids, attention_mask, bef_image, aft_image, image_mask=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None, target_ids=None):
        # The text slots hold rules (B, T), rule masks (B, T, K) and the
        # validity of the rules (dataloader_trance_grammar.py).
        rules = input_ids.view(-1, input_ids.shape[-1])
        rule_mask = token_type_ids.view(-1, *token_type_ids.shape[-2:]).bool()
        image_mask = image_mask.view(-1, image_mask.shape[-1])
        image = torch.cat([bef_image, aft_image], 1)
        b, pair, channel, h, w = image.shape
        visual_z, _ = self.get_visual_output(image.view(b * pair, channel, h, w), image_mask, shaped=True,
                                             video_frame=pair)
        if not self.training:
            return None

        if self._stage_one:
            logits, z_mean, z_log_var = self.gvae(rules)
            sim_matrix, _ = self.get_similarity_logits(z_mean, visual_z, attention_mask, image_mask, shaped=True)
            negative_mask = None if target_ids is None else self._same_target_mask(target_ids)
            sim_loss = (self.loss_fct(sim_matrix, negative_mask) + self.loss_fct(sim_matrix.T, negative_mask)) / 2
            recon = reconstruction_loss(logits, rules, rule_mask)
            kl = kl_loss(z_mean, z_log_var)
            loss = sim_loss + self.task_config.gvae_weight * (recon + self.task_config.gvae_kl_weight * kl)
            self.loss_terms = {"contrastive": float(sim_loss), "recon": float(recon), "kl": float(kl)}
        else:
            loss = reconstruction_loss(self.gvae.decode(visual_z.squeeze(1)), rules, rule_mask)
            self.loss_terms = {"recon": float(loss)}
        return loss

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        """(z_mean, z_log_var) of the derivations; z_mean takes the place of the text embedding."""
        return self.gvae.encode(input_ids.view(-1, input_ids.shape[-1]))

    def pair_latents(self, bef_image, aft_image, image_mask):
        """Latents (B, latent_dim) of image pairs, the z the GVAE decoder reads in stage 2."""
        visual_z, _ = self.get_visual_output(torch.cat([bef_image, aft_image], 1), image_mask)
        return visual_z.squeeze(1)

    def rule_logits(self, bef_image, aft_image, image_mask):
        """Unmasked rule logits (B, T, K) of image pairs, all T steps at once."""
        return self.gvae.decode(self.pair_latents(bef_image, aft_image, image_mask))

    def decode_latents(self, z, maskers, sample=False):
        """Derivations (B, T) decoded from latents (B, latent_dim) under one masker each."""
        return masked_greedy_decode(self.gvae.decode(z), maskers, sample)

    def decode_rules(self, bef_image, aft_image, image_mask, maskers, sample=False):
        """Derivations (B, T) decoded from image pairs under one masker per pair."""
        return masked_greedy_decode(self.rule_logits(bef_image, aft_image, image_mask), maskers, sample)
