import logging
import os
import sys
from typing import Tuple

# ToDo: Find a proper way to add this to the PYTHONPATH env-variable
sys.path.append('C:/Users/Tiz/PycharmProjects/DDQN_Trading_MSC')
logging.basicConfig(level=logging.INFO)

import click
from pathlib import Path

from dqn.algorithm.config_parsing.dqn_config import DqnConfig
from dqn.dqn_core import DqnGym

_DEFAULT_OUT = "../out/dqn"
_DEFAULT_CONFIG = "./hyper_parameters.json"

@click.group()
def cli():
    pass

@cli.command()
@click.option('--out', help='Output path the resulting model will be saved under.', type=click.Path(), default=_DEFAULT_OUT)
@click.option('--config', help='Config file path the hyper parameters are gathered from.', default=_DEFAULT_CONFIG)
@click.option('--model', help='Selects model to train. This can be \'latest\' or the folder name under the out path')
def train(out: str, config: str, model: str):
    abs_out, abs_config = _abs_paths(out, config)
    gym = DqnGym(abs_out, DqnConfig(abs_config))
    if model is None:
        gym.train("Google")
    else:
        agent_pickle = _resolve_agent(abs_out, model)
        gym.train("Google", old_agent=agent_pickle)

@cli.command()
@click.option('--out', help='Output path the resulting model will be saved under.', type=click.Path(), default=_DEFAULT_OUT)
@click.option('--config', help='Config file path the hyper parameters are gathered from.', default=_DEFAULT_CONFIG)
@click.option('--model', required=True, help='Selects model to train. This can be \'latest\' or the folder name under the out path')
def evaluate(out: str, config: str, model: str):
    abs_out, abs_config = _abs_paths(out, config)
    agent_pickle = _resolve_agent(abs_out, model)
    DqnGym(abs_out, DqnConfig(abs_config)).evaluate("Google", agent_file=agent_pickle)

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
            logging.error(f"Could not find fold: \'{name}\' under out path: {out_path}")
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
