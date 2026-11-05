"""Command line options of the grammar target, shared by both main scripts.

Kept apart from trance_grammar.py so that the scripts can parse them without
importing the TRANCE grammar (CLEVR and Spot runs do not need it).
"""
# The rule masks of trance_grammar.py.
MASKS = ("init_desc", "program", "cfg")
# Scenes the init_desc mask of a pair reads (dataloader_trance_grammar.py).
INIT_DESC_SCOPES = ("own", "group")
# Metrics of main_task_caption.eval_epoch_grammar that can select the best
# epoch: the match rates and those of trance_execution.execution_metrics.
GRAMMAR_SELECT_METRICS = ("exact", "valid", "valid_group", "complete", "sample", "group_all", "cross_pairs", "pair_explains_group",
                          "group_any_explains_all", "group_every_explains_all")


def add_grammar_args(parser):
    parser.add_argument("--target", default="text", choices=["text", "grammar"],
                        help="TRANCE target: text through CLIP's text tower, or the grammar derivation through a GVAE.")
    parser.add_argument("--grammar_mask", default="init_desc", choices=MASKS,
                        help="Rule mask of the GVAE loss and decoding (see modules/trance_grammar.py).")
    parser.add_argument("--init_desc_scope", default="own", choices=INIT_DESC_SCOPES,
                        help="Scenes the init_desc mask of a pair reads: own = the pair's own initial scene "
                             "(default); group = every member's scene of its transformation, which leaks the "
                             "other members' scenes and leaves a single choice of object (runs before 2026-11-03).")
    parser.add_argument("--grammar_mask_cache", default=None, type=str,
                        help="Directory for the teacher masks of each split (computed once, reused while the data, "
                             "mask, scope and grammar are unchanged); none: computed at every load.")
    parser.add_argument("--grammar_mask_workers", default=0, type=int,
                        help="Processes computing the teacher masks (0: every CPU available to the job).")
    parser.add_argument("--latent_dim", "--gvae_latent_dim", dest="latent_dim", type=int, default=56,
                        help="TRANCE: size of the latent space where targets and image pairs are compared, the "
                             "same for the text and the grammar target (GVAE latent; 0 = CLIP's space, text only).")
    parser.add_argument("--gvae_hidden", type=int, default=256, help="GRU size of the GVAE decoder.")
    parser.add_argument("--gvae_weight", type=float, default=1.0,
                        help="Stage 1: weight of the GVAE loss next to the contrastive loss.")
    parser.add_argument("--gvae_kl_weight", type=float, default=1.0, help="Weight of the KL term in the GVAE loss.")
    parser.add_argument("--gvae_epsilon_std", type=float, default=0.01,
                        help="Noise scale of the reparameterisation (0.01 in the original code).")
    parser.add_argument("--freeze_gvae_decoder", action="store_true",
                        help="Stage 2: keep the GVAE decoder fixed (weights and BatchNorm statistics), so that the "
                             "image pairs must learn the codes of the stage-1 GVAE (default: it keeps training).")
