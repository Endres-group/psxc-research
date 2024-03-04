"""
This script trains the policy of the suppression of pseudopods
"""

import argparse
from collections import namedtuple
import json
import pathlib
import time
import dataclasses
from tensorboardX import SummaryWriter

import jax
import orbax.checkpoint as ocp

import psxc

parser = argparse.ArgumentParser(description="Script to run experiment.")

# ExperimentConfig parameters
parser.add_argument("--logdir", type=str, default="./runs/")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--discrete", default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()


ec_args = {k: v for k, v in vars(args).items() if k in psxc.ppo.ExperimentConfig.__dict__}
config = psxc.ppo.ExperimentConfig(**ec_args, length=1.0)

env_params = psxc.Parameters()
key = jax.random.key(args.seed)

Checkpointer = namedtuple("Checkpointer", ["dir", "writer", "manager"])


def init_logger(config, params):
    # Create the directory with the logs and checkpoints.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_type = "discrete" if config.discrete else "continuous"
    experiment_name = f"ppo_{model_type}_{timestamp}_{config.seed}"
    model_dir = pathlib.Path(config.logdir) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialise the tensorboard logger and same the config files.
    summary_writer = SummaryWriter(model_dir.absolute())
    with open(model_dir / "env_params.json", "w") as f:
        f.write(json.dumps(dataclasses.asdict(params), indent=4))
    with open(model_dir / "config.json", "w") as f:
        f.write(json.dumps(config._asdict(), indent=4))
    print(f"Writting logs to: {model_dir}")

    manager = ocp.StandardCheckpointer()
    return Checkpointer(model_dir, summary_writer, manager)


checkpointer = init_logger(config, env_params)
psxc.ppo.train_loop(key, config, env_params, checkpointer)
