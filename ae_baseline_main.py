import torch
import numpy as np

import torch
import logging
import random
import numpy as np

from utils.config import Config
from datasets.util import load_dataset
from base.base_dataset import DatasetSplit
from trainer.ae_trainer_mvtec import  AETrainer_mvtec
import os

def main(cfg):

    data_path = cfg.settings['trainer_para']['data_path']
    dataset_name = cfg.settings['trainer_para']['dataset_name']
    class_name = cfg.settings['trainer_para']['class_name']
    net_name = cfg.settings['trainer_para']['net_name']
    outdir_final_index_name = cfg.settings['trainer_para']['outdir_final_index_name']
    # set output dir
    final_outpath_ind = 1
    export_dir = "output/train/" + dataset_name + '/' + class_name + '/'
    out_path = export_dir + class_name + '_' + net_name + "_" + outdir_final_index_name +"_" + str(final_outpath_ind)

    while(os.path.exists(out_path)):
        final_outpath_ind = final_outpath_ind + 1
        out_path = export_dir + class_name + '_' + net_name + "_" + outdir_final_index_name + "_" + str(final_outpath_ind)
    os.makedirs(out_path)

    # Set up logging
    log_file = out_path + '/log.txt'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % out_path)
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Trained class: %s' % class_name)
    logger.info('Network: %s' % net_name)

    # Set seed
    if cfg.settings['trainer_para']['seed'] != -1:
        random.seed(cfg.settings['trainer_para']['seed'])
        np.random.seed(cfg.settings['trainer_para']['seed'])
        torch.manual_seed(cfg.settings['trainer_para']['seed'])
        logger.info('Set seed to %d.' % cfg.settings['trainer_para']['seed'])

    logger.info('Computation device: %s' % cfg.settings['trainer_para']['device'])

    # Load data
    image_size = cfg.settings["trainer_para"]['image_size']
    train_dataset = load_dataset(dataset_name, data_path, image_size, class_name, task=DatasetSplit.TRAIN)
    test_dataset  = load_dataset(dataset_name, data_path, image_size, class_name, task=DatasetSplit.TEST)

    deep_trainer = AETrainer_mvtec(
                                    trainer_para=cfg.settings['trainer_para'],
                                    optim_para=cfg.settings['optim_para'],
                                    dataloader_para=cfg.settings['dataloader_para'],
                                    scheduler_para=cfg.settings['scheduler_para'],
                                    network_init_para = cfg.settings['network_init_para'],
                                    out_path = out_path,
                                    mode = "train"
                                    )

    # Train model on dataset
    deep_trainer.train(train_dataset)

    # Test model
    deep_trainer.test(test_dataset)

    deep_trainer.save_results()

    cfg.save_config(out_path)

if __name__ == '__main__':

    
    load_config_path = "jsons/ae_mvtec_loco.json"
    cfg = Config()
    cfg.load_config(import_json=load_config_path)

   
    for item in cfg.settings['trainer_para']["class_list"]:
        cfg.load_config(import_json=load_config_path)
        cfg.settings['trainer_para']["class_name"] = item
        main(cfg)
