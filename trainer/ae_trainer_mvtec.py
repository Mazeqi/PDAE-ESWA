from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet

import logging
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import json
from utils import metrics
import cv2
from utils.utils import calculate_resmaps, plot_reconstruct_images, write_results
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from networks.util import RescaleSegmentor

class AETrainer_mvtec(BaseTrainer):

    def __init__(self, trainer_para = {},  optim_para={}, scheduler_para={}, dataloader_para={}, network_init_para={}, out_path="", mode = "train"):
        super().__init__(trainer_para, optim_para, scheduler_para, dataloader_para, network_init_para, out_path, mode)
        
    def train(self, dataset: Dataset):
        logger = logging.getLogger()

        # Get train data loader
        train_loader = DataLoader(dataset=dataset, shuffle=True, **self.dataloader_para)
        
        # Set optimizer (Adam optimizer for now)
        optimizer = self.optimizer

        # Set learning rate scheduler
        scheduler = self.scheduler

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        self.net.train()
        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            with tqdm.tqdm(train_loader, desc="Train Model...", position=1, leave=False
                           ) as data_iterator:
                for inputs in data_iterator:
                    if isinstance(inputs, dict):
                        inputs = inputs["image"]

                    inputs = inputs.to(self.device)

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = self.net(inputs)
                    
                    loss = self.criterion(inputs, outputs)
                    loss.backward()
                    optimizer.step()
                    
                    loss_epoch += loss.item()
                    n_batches += 1

            if self.scheduler_name == "plateau":        
                scheduler.step(loss)
            else:
                scheduler.step()
                
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

            if epoch % 5 == 0:
                with torch.no_grad():
                    array_inputs  = inputs[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_outputs = outputs[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)

                    plot_reconstruct_images(self.out_path + "/train_images/", array_inputs, array_outputs)
                    self.save_network(self.net, "CAE")

           
        train_time = time.time() - start_time
        logger.info('Train time: %.3f' % train_time)
        logger.info('Finished pretraining.')
        self.results["train_time"] = train_time

        return self.net

    def test(self, dataset: Dataset):
        logger = logging.getLogger()

        # Get test data loader
        test_loader = DataLoader(dataset=dataset, shuffle=False, **self.dataloader_para)
        
        # Testing
        logger.info('Testing autoencoder...')
        start_time = time.time()
        self.net.eval()

        mask_gt = []
        dist_maps = []
        ori_imgs = []
        rec_imgs = []
        images_scores = []
        images_labels = []
        with torch.no_grad():
            with tqdm.tqdm(test_loader, desc="Val Model...", position=1, leave=False
                           ) as data_iterator:
                
                for data in data_iterator:
                    inputs, masks, labels = data["image"], data["mask"], data['label']
                    inputs = inputs.to(self.device)
                    outputs = self.net(inputs)

                    array_inputs  = inputs.cpu().numpy().transpose(0, 2, 3, 1)
                    array_outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
                    if self.loss_name == "mse":
                        scores, remaps = calculate_resmaps(array_inputs, array_outputs, "l2")
                    else:
                        scores, remaps = calculate_resmaps(array_inputs, array_outputs, "ssim")

                    # transpose for display
                    mask_gt.extend([iMap.squeeze() for iMap in masks.cpu().numpy()])
                    dist_maps.extend([iMap  for iMap in remaps])
                    ori_imgs.extend([iMap for iMap in array_inputs])
                    rec_imgs.extend([iMap for iMap in array_outputs])
                    images_scores.extend(scores)
                    images_labels.extend(labels.tolist())
        
        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
        rescaleSegmentor = RescaleSegmentor(self.device, target_size=self.trainer_para['image_size'])
        dist_maps = np.array(dist_maps)
        print(dist_maps.shape)
        dist_maps = rescaleSegmentor.convert_to_segmentation(dist_maps)

        dist_maps = np.array(dist_maps)
        max_score = dist_maps.max()
        min_score = dist_maps.min()
        dist_maps = (dist_maps - min_score) / (max_score - min_score)

        images_scores = np.array(images_scores)
        max_score = images_scores.max()
        min_score = images_scores.min()
        images_scores = (images_scores - min_score) / (max_score - min_score)

        #plot_segmentation_images(self.out_path + "/test_images/", ori_images=ori_imgs, rec_imgs = rec_imgs, mask_gts=mask_gt, mask_pred=dist_maps)
           
        image_test_results = metrics.compute_imagewise_retrieval_metrics(
            images_scores, images_labels
        )

        pix_test_results = metrics.compute_pixelwise_retrieval_metrics(
                dist_maps, mask_gt
        )

        # Compute PRO score & PW Auroc only images with anomalies
        sel_idxs = []
        for i in range(len(mask_gt)):
            if np.sum(mask_gt[i]) > 0:
                sel_idxs.append(i)
        anomaly_pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            [dist_maps[i] for i in sel_idxs],
            [mask_gt[i] for i in sel_idxs],
        )
        anomaly_pixel_auroc = anomaly_pixel_scores["auroc"]
        
        
        logger.info('Test set image AUC: {:.2f}%'.format(100. * image_test_results["auroc"]))
        logger.info('Test set pix AUC: {:.2f}%'.format(100. * pix_test_results["auroc"]))
        logger.info('Test set anomaly pix AUC: {:.2f}%'.format(100. * anomaly_pixel_scores["auroc"]))
        self.results['test_time'] = test_time

        self.results['test_results'] = {
            'image_auroc':image_test_results['auroc'] * 100. ,
            'pixel_auroc':pix_test_results['auroc'] * 100. ,
            'pix_pro':anomaly_pixel_auroc * 100. 
        }

        if self.mode != "test":
            out_path = "output/train/"
        else:
            out_path = "output/test/"

        write_results(
            self.results['test_results'],
            self.trainer_para["class_name"],
            self.trainer_para["class_list"],
            out_path +"/"+ self.trainer_para["dataset_name"] + "/result.csv"
        )

    
    
    