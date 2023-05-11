import click

from dqn_legacy_code.ModelExecution.ModelEvaluator import ModelEvaluator
from dqn_legacy_code.ModelExecution.ModelServer import ModelServer
from dqn_legacy_code.ModelTraining.model_trainer import ModelTrainer
from dqn_legacy_code.ConfigClasses.model_config import ModelConfig

@click.command()
@click.option('--config', default='./model_config.json')
def run(config: str):

    model_config = ModelConfig(config)
    if model_config.execution_type == 0:
        print(f"[{model_config.execution_type=}] Selected model training.")
        ModelTrainer(model_config).start()
    elif model_config.execution_type == 1:
        print(f"[{model_config.execution_type=}] Selected model evaluation.")
        ModelEvaluator(model_config).evaluate()
    elif model_config.execution_type == 2:
        print(f"[{model_config.execution_type=}] Selected model evaluation.")
        ModelServer(model_config).serve()
    else:
        print(f"[{model_config.execution_type=}] Unsupported option.")
        pass

if __name__ == '__main__':
    run()
