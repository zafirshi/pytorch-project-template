from loguru import logger

from project import Experiment
from project.config import config
from project.models import models

import torch
from pytorch_lightning import Trainer
test = torch.randn(1, 3, 32, 32)

model_cls = models[config.model_config.name]
model = model_cls(**config.model_config.dict())
logger.info("Loaded")

# 训练和测试的代码可以分别在project文件夹中分两个Train和Test脚本写两个类

# 模型训练输出可以放在project下的Code_output文件中，
# --Code_output
#   --run
#     --tensorboardlog
#       - log
#   --save
#     - checkpoint.pth
#     - flag.json # 这次运行的配置，方便重加载和对比
expt = Experiment(model=model, **config.experiment_config.dict())
trainer = Trainer(**config.trainer_config.dict())
trainer.fit(expt)
