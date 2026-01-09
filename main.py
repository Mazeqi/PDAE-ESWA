import torch
import numpy as np

import torch
import logging
import random
import numpy as np

from utils.config import Config
from datasets.util import load_dataset
from base.base_dataset import DatasetSplit
import os
from trainer.multi_back_bone_trainer import MultiBackBoneTrainer

def main(cfg):

    # Get configuration
    
    data_path = cfg.settings["trainer_para"]['data_path']
    dataset_name = cfg.settings["trainer_para"]['dataset_name']
    class_name = cfg.settings["trainer_para"]['class_name']
    net_name = cfg.settings['trainer_para']['net_name']
    outdir_final_index_name = cfg.settings['trainer_para']['outdir_final_index_name']

    # set output dir
    final_outpath_ind = 1
    export_dir = "output/train/" + dataset_name + '/' + class_name + '/'
    out_path = export_dir + class_name + "_" + outdir_final_index_name +"_" + str(final_outpath_ind)

    while(os.path.exists(out_path)):
        final_outpath_ind = final_outpath_ind + 1
        out_path = export_dir + class_name + "_" + outdir_final_index_name + "_" + str(final_outpath_ind)
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

    # Print argumentstrain
    #logger.info('Loaded configuration from %s.' % load_config)
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % out_path)
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Trained class: %s' % class_name)
    logger.info('Network: %s' % net_name)
    
    # Set seed
    if cfg.settings["trainer_para"]['seed'] != -1:
        random.seed(cfg.settings["trainer_para"]['seed'])
        np.random.seed(cfg.settings["trainer_para"]['seed'])
        torch.manual_seed(cfg.settings["trainer_para"]['seed'])
        torch.cuda.manual_seed_all(cfg.settings["trainer_para"]['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings["trainer_para"]['seed'])

    logger.info('Computation device: %s' % cfg.settings['trainer_para']['device'])
    # Load data

    image_size = cfg.settings["trainer_para"]['image_size']
    train_dataset = load_dataset(dataset_name, data_path, image_size, class_name, task=DatasetSplit.TRAIN, dataset_para = cfg.settings["dataset_para"])
    test_dataset  = load_dataset(dataset_name, data_path, image_size, class_name, task=DatasetSplit.TEST,  dataset_para = cfg.settings["dataset_para"])
    
    deep_trainer = MultiBackBoneTrainer(
                                    trainer_para=cfg.settings['trainer_para'],
                                    optim_para=cfg.settings['optim_para'],
                                    dataloader_para=cfg.settings['dataloader_para'],
                                    scheduler_para=cfg.settings['scheduler_para'],
                                    network_init_para = cfg.settings['network_init_para'],
                                    out_path = out_path,
                                    mode = cfg.settings['trainer_para']["train_mode"],
                                    dataset_para=cfg.settings["dataset_para"]
                                    )

    cfg.save_config(out_path)
    
    # Train model on dataset
    deep_trainer.train(train_dataset, test_dataset)

    # Test model
    #deep_trainer.test(deep_trainer.aug_model, test_dataset)

    #deep_trainer.save_results()

    cfg.save_config(out_path)

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    from shutil import copy
    net_yaml_name = cfg.settings['trainer_para']['net_name']
    copy("yamls/" + net_yaml_name +".yaml", out_path + "/" + net_yaml_name + ".yaml")
    
if __name__ == '__main__':

    load_config_path = "jsons/mvtec_loco_aug_methods_multi_bb.json"
    cfg = Config()
    cfg.load_config(import_json=load_config_path)

    if cfg.settings["trainer_para"]["train_class_mode"] == "one_class":
        cfg.load_config(import_json=load_config_path)
        main(cfg)

    elif cfg.settings["trainer_para"]["train_class_mode"] == "one_class_para":
        for item in range(0, 20):
            cfg.load_config(import_json=load_config_path)
            beta_contras = cfg.settings["trainer_para"]["loss_contrastive_beta"]
            cfg.settings["trainer_para"]["loss_contrastive_beta"] += (beta_contras * (10**item))
            main(cfg)

    elif cfg.settings["trainer_para"]["train_class_mode"] == "all_class":
        #-------------train all class
        # Get configuration
        for item in cfg.settings["trainer_para"]["class_list"]:
            cfg.load_config(import_json=load_config_path)
            cfg.settings["trainer_para"]["class_name"] = item
            main(cfg)
