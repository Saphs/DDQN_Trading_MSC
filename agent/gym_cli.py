import logging
import os
import sys
from typing import Tuple
import click
from pathlib import Path

# ToDo: Find a proper way to add this to the PYTHONPATH env-variable
sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.gym_core import DqnGym, set_seed

_DEFAULT_OUT = "../out/agent"
_DEFAULT_CONFIG = "./hyper_parameters.json"
_DEFAULT_SEED = 1234567890


@click.group()
@click.option('-o', '--out', help='Output path the resulting model will be saved under.', type=click.Path(), default=_DEFAULT_OUT)
@click.option('-c', '--config', help='Config file path the hyper parameters are gathered from.', default=_DEFAULT_CONFIG)
@click.option('-s', '--seed', help='Seed for any nondeterministic actions.', type=int, default=_DEFAULT_SEED)
@click.pass_context
def cli(ctx: click.Context, out: str, config: str, seed: int):
    ctx.ensure_object(dict)
    set_seed(seed)
    abs_out, abs_config = _abs_paths(out, config)
    ctx.obj['abs_out'] = abs_out
    ctx.obj['config'] = DqnConfig(abs_config)


@cli.command()
@click.option('-m', '--model', help='Selects model to train. This can be \'latest\' or the folder name under the out path')
@click.pass_context
def train(ctx: click.Context, model: str):
    abs_out, config = ctx.obj.values()
    gym = DqnGym(abs_out, config)
    if model is None:
        gym.train(config.environment.data_set_name)
    else:
        agent_pickle = _resolve_agent(abs_out, model)
        gym.train(config.environment.data_set_name, old_agent=agent_pickle)


@cli.command()
@click.option('--model', required=True, help='Selects model to train. This can be \'latest\' or the folder name under the out path')
@click.pass_context
def evaluate(ctx: click.Context, model: str):
    abs_out, config = ctx.obj.values()
    agent_pickle = _resolve_agent(abs_out, model)
    DqnGym(abs_out, config).evaluate(config.environment.data_set_name, agent_file=agent_pickle)


# -- Helper functions --
def _abs_paths(out: str, config: str) -> Tuple[Path, Path]:
    # These might be a string and not a Path due to weird click CLI typing, we force it into a Path
    abs_config = Path(os.path.join(Path(__file__).parent.resolve(), config))
    abs_out = Path(os.path.join(Path(__file__).parent.resolve(), out))
    abs_out.mkdir(parents=True, exist_ok=True)
    return abs_out, abs_config


def _resolve_agent(out_path: Path, name: str) -> Path:
    if name == 'latest':
        model_folder = _find_latest_folder(out_path)
        logging.info(f"Latest model found in folder: {model_folder}")
    else:
        model_folder = _find_named_folder(out_path, name)
    return _model_file_in_folder(model_folder)


def _find_named_folder(out_path: Path, name: str) -> Path:
    if name.isnumeric():
        numeric_folder_names = list(filter(lambda s: s.isnumeric(), os.listdir(out_path)))
        if name in numeric_folder_names:
            return Path(os.path.abspath(os.path.join(str(out_path), name)))
        else:
            logging.error(f"Could not find folder: \'{name}\' under out path: {out_path}")
            sys.exit(1)
    else:
        logging.error(f"Agent folder should be numeric: \'{name}\' is not numeric")
        sys.exit(1)


def _find_latest_folder(out_path: Path) -> Path:
    numeric_folder_names = list(filter(lambda s: s.isnumeric(), os.listdir(out_path)))
    numeric_folder_names.sort()
    return Path(os.path.abspath(os.path.join(str(out_path), str(numeric_folder_names[-1]))))


def _model_file_in_folder(folder: Path) -> Path:
    folder_content = os.listdir(folder)
    try:
        agent_pickle = list(filter(lambda s: s.endswith(".pkl") and s.startswith("model_"), folder_content))[0]
        return Path(os.path.join(folder, agent_pickle))
    except IndexError as err:
        logging.error(err)
        logging.error(f"Could not find model file (.pkl) under {folder}.")
        sys.exit(1)

if __name__ == '__main__':
    cli()
