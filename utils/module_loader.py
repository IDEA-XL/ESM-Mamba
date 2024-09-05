import os
import copy
import pytorch_lightning as pl
import datetime

import wandb
from pytorch_lightning.loggers import TensorBoardLogger
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from pytorch_lightning.strategies import DDPStrategy


def load_tensorboard(config):
    # initialize tensorboard
    tensorboard_config = config.setting.tensorboard
    tensorboard_logger = TensorBoardLogger(tensorboard_config.log_dir, 
                                           name=tensorboard_config.name,
                                           version=tensorboard_config.version if "version" in tensorboard_config else None)
    return tensorboard_logger


def load_model(config):
    # initialize model
    model_config = copy.deepcopy(config)
    kwargs = model_config.pop('kwargs')
    model_config.update(kwargs)
    return ModelInterface.init_model(**model_config)


def load_dataset(config):
    # initialize dataset
    dataset_config = copy.deepcopy(config)
    kwargs = dataset_config.pop('kwargs')
    dataset_config.update(kwargs)
    return DataInterface.init_dataset(**dataset_config)


# Initialize strategy
def load_strategy(config):
    config = copy.deepcopy(config)
    if "timeout" in config.keys():
        timeout = int(config.pop('timeout'))
        config["timeout"] = datetime.timedelta(seconds=timeout)

    return DDPStrategy(**config)


# Initialize a pytorch lightning trainer
def load_trainer(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Initialize tensorboard logger
    if trainer_config.logger:
        trainer_config.logger = load_tensorboard(config)
    else:
        trainer_config.logger = False
    
    # Initialize strategy
    strategy = load_strategy(trainer_config.pop('strategy'))
    return pl.Trainer(**trainer_config, strategy=strategy, callbacks=[])