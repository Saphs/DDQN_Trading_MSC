import logging
import os
import sys
from typing import Tuple
import click
from pathlib import Path



# ToDo: Find a proper way to add this to the PYTHONPATH env-variable
sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from agent.algorithm.config_parsing.dqn_config_codec import DqnConfigCodec
from agent.gym_core import DqnGym, set_seed

_DEFAULT_OUT = "../out/agent"
_DEFAULT_CONFIG = "./hyper_parameters.json"
_DEFAULT_SEED = 1234567890


@click.group()
@click.option('-o', '--out', help='Output path the resulting model will be saved under.', type=click.Path(), default=_DEFAULT_OUT)
@click.option('-c', '--config', help='Config file path the hyper parameters are gathered from.', default=_DEFAULT_CONFIG)
@click.pass_context
def cli(ctx: click.Context, out: str, config: str):
    ctx.ensure_object(dict)
    abs_out, abs_config = _abs_paths(out, config)
    ctx.obj['abs_out'] = abs_out
    ctx.obj['config'] = DqnConfigCodec.read_json(abs_config)
    if ctx.obj['config'].seed == "default":
        set_seed(_DEFAULT_SEED)
    else:
        set_seed(ctx.obj['config'].seed)
    logging.info(f"Using config: {ctx.obj['config']}")


@cli.command()
@click.option('-m', '--model', help='Selects model to train. This can be \'latest\' or the folder name under the out path')
@click.option('-n', '--named', help='Provides name the agent will be named to.')
@click.pass_context
def train(ctx: click.Context, model: str, named: str):
    abs_out, config = ctx.obj.values()
    gym = DqnGym(abs_out, config)
    if model is None:
        gym.train(config.environment.data_set_name, name=named)
    else:
        agent_pickle = _resolve_agent(abs_out, model)
        gym.train(config.environment.data_set_name, old_agent=agent_pickle)

    agent_pickle = _resolve_agent(abs_out, gym.result_path.name)
    gym.evaluate(config.environment.data_set_name, agent_file=agent_pickle)


@cli.command()
@click.option('--model', required=True, help='Selects model to train. This can be \'latest\' or the folder name under the out path')
@click.pass_context
def evaluate(ctx: click.Context, model: str):
    abs_out, config = ctx.obj.values()
    agent_pickle = _resolve_agent(abs_out, model)
    DqnGym(abs_out, config).evaluate(config.environment.data_set_name, agent_file=agent_pickle)

@cli.command()
@click.option('--model', required=True, help='Selects model to train. This can be \'latest\' or the folder name under the out path')
@click.pass_context
def evaluate_checkpoints(ctx: click.Context, model: str):
    abs_out, config = ctx.obj.values()
    checkpoints = _resolve_checkpoints(abs_out, model)
    DqnGym(abs_out, config).check_point_eval(config.environment.data_set_name, checkpoints=checkpoints)

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


def _resolve_checkpoints(out_path: Path, name: str) -> list[Path]:
    if name == 'latest':
        model_folder = _find_latest_folder(out_path)
        logging.info(f"Latest model found in folder: {model_folder}")
    else:
        model_folder = _find_named_folder(out_path, name)

    checkpoint_folder = model_folder.joinpath("checkpoints")
    path_strs = os.listdir(checkpoint_folder)
    pickles = list(filter(lambda s: s.endswith(".pkl") and s.startswith("model_"), path_strs))
    pickles = list(map(lambda p: checkpoint_folder.joinpath(p), pickles))
    return pickles


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
    result_folder = folder.joinpath("final")
    folder_content = os.listdir(result_folder)
    try:
        agent_pickle = list(filter(lambda s: s.endswith(".pkl") and s.startswith("best_"), folder_content))[0]
        return result_folder.joinpath(agent_pickle)
    except IndexError as err:
        logging.error(err)
        logging.error(f"Could not find model file (.pkl) under {folder}.")
        sys.exit(1)

if __name__ == '__main__':
    cli()
