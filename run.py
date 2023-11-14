import click

from src.tools.hyperparameters_tuning import get_best_params
from src.tools.train_model import train_model
from src.tools.predict import single_prediction


@click.group()
def cli():
    pass


cli.add_command(get_best_params)
cli.add_command(train_model)
cli.add_command(single_prediction)

if __name__ == "__main__":
    cli()
