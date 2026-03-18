import logging
import os


_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_FORMAT, level=log_level)


def log_to_file(logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"):
    log_dir = os.path.dirname(log_filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log = logging.getLogger(logger_name) if logger_name is not None else logging.getLogger()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    log.addHandler(file_handler)


def log_versions():
    import subprocess
    import torch

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        logging.info("git hash: %s", git_hash)
    except Exception:
        logging.info("git hash: unavailable")

    logging.info("Torch: %s", torch.__version__)

