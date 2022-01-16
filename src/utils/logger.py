import os
from datetime import datetime
import logging


def get_logger(args):
    stamp = datetime.now().strftime('%Y%m%d/')
    log_folder = os.path.join("./logs/", stamp, args.dataset)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = log_folder + '/%s.log' % stamp
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    return logger, stamp
