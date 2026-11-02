"""Command line options of the grammar target, shared by both main scripts.

Kept apart from trance_grammar.py so that the scripts can parse them without
importing the TRANCE grammar (CLEVR and Spot runs do not need it).
"""
# The rule masks of trance_grammar.py.
MASKS = ("init_desc", "program", "cfg")


def add_grammar_args(parser):
    parser.add_argument("--target", default="text", choices=["text", "grammar"],
                        help="TRANCE target: text through CLIP's text tower, or the grammar derivation through a GVAE.")
    parser.add_argument("--grammar_mask", default="init_desc", choices=MASKS,
                        help="Rule mask of the GVAE loss and decoding (see modules/trance_grammar.py).")
    parser.add_argument("--latent_dim", "--gvae_latent_dim", dest="latent_dim", type=int, default=56,
                        help="TRANCE: size of the latent space where targets and image pairs are compared, the "
                             "same for the text and the grammar target (GVAE latent; 0 = CLIP's space, text only).")
    parser.add_argument("--gvae_hidden", type=int, default=256, help="GRU size of the GVAE decoder.")
    parser.add_argument("--gvae_weight", type=float, default=1.0,
                        help="Stage 1: weight of the GVAE loss next to the contrastive loss.")
    parser.add_argument("--gvae_kl_weight", type=float, default=1.0, help="Weight of the KL term in the GVAE loss.")
    parser.add_argument("--gvae_epsilon_std", type=float, default=0.01,
                        help="Noise scale of the reparameterisation (0.01 in the original code).")
