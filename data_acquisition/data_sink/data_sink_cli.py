import logging
import os
import sys
import click
logging.basicConfig(level=logging.INFO)


# ToDo: Find a proper way to add this to the PYTHONPATH env-variable
sys.path.append(os.getcwd())
from data_acquisition.data_sink.data_sink_core import DataSink


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', help='Source file to load the data from', type=click.File('rb'))
@click.option('--name', help="Name the stock data is saved as.", type=str)
@click.option('--symbol', help="Stock trading symbol the stock is known as.", type=str)
def import_file(source: click.File, name: str, symbol: str):
    if source is not None:
        sink = DataSink()
        sink.import_from_file(source, name, symbol)
    else:
        logging.warning(f"Source argument was empty or the file was not found. Be sure to provide a source CSV.")


@cli.command()
def list():
    summary = DataSink().list()
    print(summary)


@cli.command()
def clean():
    DataSink().clean()


@cli.command()
@click.option('--name', help="Name the stock data was imported under.", type=str)
@click.option('--identifier', help="DataBase ID of the stock data.", type=int)
def plot(name: str, identifier: int):
    DataSink().plot(name, identifier)


@cli.command()
@click.option('--old', required=True, help="Old name for the stock that is to be renamed.", type=str)
@click.option('--new', required=True, help="The new name that will override the old one.", type=str)
def rename(old: str, new: str):
    DataSink().rename(old, new)


if __name__ == '__main__':
    cli()

