import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import logging

def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

# =====================================================================
# Resumable training
# =====================================================================
# Every epoch saves the weights to pytorch_model.bin.<epoch> and, after the
# epoch's evaluation, the training state to pytorch_opt.bin.<epoch>: optimizer
# (which holds the schedule's step count), counters, best score and RNG states.
# The state file is written last, so its presence marks a complete epoch.


def atomic_save(obj, path):
    """torch.save through a temporary file, so a killed job never leaves a partial file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def resolve_resume(args):
    """The training state to resume from, or None.

    --resume_model is a pytorch_opt.bin.<epoch> file, or "latest": the newest
    one in --output_dir whose weights exist (None if there is none, so a job
    can always pass "latest"). The weights of that epoch replace --init_model.
    """
    if not args.resume_model:
        return None
    path = args.resume_model
    if path == "latest":
        done = []
        for state_file in glob.glob(os.path.join(args.output_dir, "pytorch_opt.bin.*")):
            epoch = state_file.rsplit(".", 1)[1]
            model_file = os.path.join(args.output_dir, "pytorch_model.bin." + epoch)
            if epoch.isdigit() and os.path.isfile(model_file):
                done.append((int(epoch), state_file))
        if not done:
            return None
        path = max(done)[1]
    # Our own files; they hold RNG states, which weights_only=True rejects.
    state = torch.load(path, map_location="cpu", weights_only=False)
    args.init_model = os.path.join(os.path.dirname(path), "pytorch_model.bin.{}".format(state["epoch"]))
    state["path"] = path
    return state


def restore_training_state(state, optimizer, num_train_optimization_steps):
    """Loads the optimizer and RNG states. Returns the first epoch to train."""
    # The schedule depends on the total number of steps, so it must not change.
    saved_total = state["optimizer_state_dict"]["param_groups"][0]["t_total"]
    if saved_total != num_train_optimization_steps:
        raise ValueError("cannot resume {}: it was trained for {} steps in total, this run has {} "
                         "(same --epochs, --batch_size and data needed)".format(
                             state["path"], saved_total, num_train_optimization_steps))
    optimizer.load_state_dict(state["optimizer_state_dict"])
    set_rng_state(state["rng"])
    return state["epoch"] + 1


def parse_args_with_config(parser, section):
    """Parse the command line, with defaults from the YAML file given by --config.

    The file has a `common` mapping and one per stage: `adaptation`
    (main_task_retrieval.py) and `finetuning` (main_task_caption.py). The
    stage's keys override `common`, the command line overrides both. Keys are
    argument names (dest); unknown keys or sections are an error.
    """
    # --config is read first on its own, so that required arguments do not fail here.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    config_path = pre.parse_known_args()[0].config
    if config_path:
        import yaml
        with open(config_path) as fp:
            config = yaml.safe_load(fp) or {}
        unknown = sorted(set(config) - {"common", "adaptation", "finetuning"})
        if unknown:
            parser.error("{}: unknown sections {}".format(config_path, unknown))
        values = dict(config.get("common") or {})
        values.update(config.get(section) or {})
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(values) - set(actions))
        if unknown:
            parser.error("{}: unknown keys {}".format(config_path, unknown))
        # YAML may read e.g. 1e-4 as a string; convert like the command line would.
        for key, value in values.items():
            if actions[key].type is not None and value is not None and not isinstance(value, bool):
                values[key] = actions[key].type(value)
        parser.set_defaults(**values)
    return parser.parse_args()


def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def log_loss_terms(logger, model):
    # the grammar model keeps the terms of its last loss (modules/modeling_grammar.py)
    loss_terms = getattr(model.module if hasattr(model, 'module') else model, "loss_terms", None)
    if loss_terms:
        logger.info("\tLoss terms: %s", ", ".join("{}: {:f}".format(k, v) for k, v in loss_terms.items()))
