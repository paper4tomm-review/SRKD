import logging
import os
import sys
import os.path as op


def setup_logger(name, save_dir, if_train, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 禁用日志传播

    # # don't log results for the non-master process
    # if distributed_rank > 0:
    #     return logger
    # 非主进程直接返回空处理器logger
    if distributed_rank > 0:
        logger.handlers = []  # 清除所有处理器
        logger.addHandler(logging.NullHandler())  # 添加空处理器
        return logger

    # 主进程配置日志处理器
    logger.handlers = []  # 清除已有处理器

    # 控制台日志
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 创建目录（仅主进程）
    if not op.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 文件日志
    if if_train:
        log_file = "train_log.txt"
    else:
        log_file = "test_log.txt"

    fh = logging.FileHandler(os.path.join(save_dir, log_file), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger