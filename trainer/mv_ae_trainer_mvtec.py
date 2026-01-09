from base.base_trainer import BaseTrainer

import logging
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
from utils import metrics
from utils.utils import plot_images
from networks.util import NetworkFeatureAggregator, RescaleSegmentor
from networks.backbones import load_backbone
import torch.nn.functional as F
from networks.CDO_loss import CDO_LOSS

class MV_AE_BASE_Trainer(BaseTrainer):

    def __init__(self, trainer_para = {},  optim_para={}, scheduler_para={}, dataloader_para={}, network_init_para={}, out_path="", mode = "train"):
        
        
        self.backbone = load_backbone(trainer_para["backbone_name"])
        self.feat_agg = NetworkFeatureAggregator(self.backbone, trainer_para["extract_layer"], trainer_para["device"])
        self.feat_agg.eval()

        rand_vec = torch.rand([1,3, 256, 256]).to(trainer_para["device"])
        output = self.feat_agg(rand_vec)

        in_ch = 0
        self.agg_bi_bigger_size = 0
        for ilayer_name in trainer_para["extract_layer"]:
            in_ch = in_ch + output[ilayer_name].shape[1]
            if output[ilayer_name].shape[2] > self.agg_bi_bigger_size:
                self.agg_bi_bigger_size = output[ilayer_name].shape[2]

        trainer_para["in_ch"] = in_ch
        trainer_para["final_ch"] = in_ch
        self.rescaleSegmentor = RescaleSegmentor(trainer_para["device"], target_size=trainer_para['image_size'], with_gaussion=True)

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
                        ori_img = inputs["image"]
                        aug_img = inputs["aug_image"]
                        aug_mask = inputs["anomaly_mask"]

                    ori_img = ori_img.to(self.device)
                    aug_img = aug_img.to(self.device)
                    with torch.no_grad():
                        out_agg_ori = self.feat_agg(ori_img)
                        out_agg_aug = self.feat_agg(aug_img)                  

                    ori_layer_list = []
                    aug_layer_list = []

                    for ilayer_name in self.trainer_para["extract_layer"]:
                        ori_layer = out_agg_ori[ilayer_name]
                        aug_layer = out_agg_aug[ilayer_name]

                        if ori_layer.shape[2] != self.agg_bi_bigger_size:
                            ori_layer = F.interpolate(
                                ori_layer,
                                size=(self.agg_bi_bigger_size, self.agg_bi_bigger_size),
                                mode="bilinear",
                                align_corners=True,
                            ) 

                            aug_layer = F.interpolate(
                                aug_layer,
                                size=(self.agg_bi_bigger_size, self.agg_bi_bigger_size),
                                mode="bilinear",
                                align_corners=True,
                            )       
                        
                        ori_layer_list.append(ori_layer)
                        aug_layer_list.append(aug_layer)

                    ori_input = torch.cat(ori_layer_list, 1)
                    aug_input = torch.cat(aug_layer_list, 1)     

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = self.net(aug_input)
                    if self.loss_name == "CDO":
                        loss = self.criterion(ori_input, outputs, aug_mask)
                    else:
                        loss = self.criterion(ori_input, outputs)
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

            if (epoch) % 5 == 0:
                with torch.no_grad():
                    array_inputs  = aug_input[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_outputs = outputs[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_aug_mask = aug_mask[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    show_ori_img  = ori_img[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    show_agu_img  = aug_img[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    
                    if self.loss_name == "mse":
                        remaps_ori = (array_inputs - array_outputs)**2
                        dis_maps = np.sqrt(np.sum(remaps_ori, axis=3))
                    else:
                        dis_maps, _ = self.criterion.cal_am(aug_input[:1,:, :, :], outputs[:1,:, :, :])

                    images_dict = {
                        "ori_img":show_ori_img,
                        "aug_img":show_agu_img,
                        "array_aug_mask":array_aug_mask,
                        "agg_input":np.sum(array_inputs, axis=3),
                        "agg_output":np.sum(array_outputs, axis=3),
                        "dis_maps":dis_maps
                    }
                    plot_images(self.out_path + "/train_images/", images_dict, denorm=True)

            if (epoch+1) % 5 == 0:
                self.save_network()
        
        train_time = time.time() - start_time
        logger.info('Train time: %.3f' % train_time)
        logger.info('Finished pretraining.')
        self.results["train_time"] = train_time

    def test(self, dataset: Dataset):
        logger = logging.getLogger()

        # Get test data loader
        test_loader = DataLoader(dataset=dataset, shuffle=False, **self.dataloader_para)
        
        # Testing
        logger.info('Testing autoencoder...')
        start_time = time.time()
        

        mask_gt_list = []
        dist_maps_list = []
        ori_imgs_list = []
        images_scores_list = []
        images_labels_list = []
        self.net.eval()
        with torch.no_grad():
            with tqdm.tqdm(test_loader, desc="Val Model...", position=1, leave=False
                           ) as data_iterator:
                
                for data in data_iterator:
                    input_imgs, masks, labels = data["image"], data["mask"], data['label']
                    input_imgs = input_imgs.to(self.device)

                    img_agg = self.feat_agg(input_imgs)
                    
                    img_agg_layer_list = []
            
                    for ilayer_name in self.trainer_para["extract_layer"]:
                        img_layer = img_agg[ilayer_name]
                        if img_layer.shape[2] != self.agg_bi_bigger_size:

                            img_layer = F.interpolate(
                                img_layer,
                                size=(self.agg_bi_bigger_size, self.agg_bi_bigger_size),
                                mode="bilinear",
                                align_corners=True,
                            ) 
                          
                        img_agg_layer_list.append(img_layer)

                    img_agg_input = torch.cat(img_agg_layer_list, 1)

                    outputs = self.net(img_agg_input)
           
                    if self.loss_name == "mse":
                        array_inputs  = img_agg_input.cpu().numpy()
                        array_outputs = outputs.cpu().numpy()
                        remaps_ori = (array_inputs - array_outputs)**2
                        remaps = np.sqrt(np.mean(remaps_ori, axis=1))
                        scores = np.sqrt(np.mean(np.max(remaps_ori, axis=(2, 3)), axis=1))

                    elif self.loss_name == "CDO":
                        cal_map = CDO_LOSS(OOM = False, gamma = self.trainer_para["CDO_GAMMA"])
                        remaps, scores = cal_map.cal_am(img_agg_input, outputs)
                        
                        
                    # transpose for display
                    mask_gt_list.extend([i_mask.squeeze() for i_mask in masks.cpu().numpy()])
                    dist_maps_list.extend([iMap  for iMap in remaps])
                    ori_imgs_list.extend([i_ori_img for i_ori_img in input_imgs.cpu().numpy().transpose(0, 2, 3, 1)])
                    images_scores_list.extend(scores)
                    images_labels_list.extend(labels.tolist())
        
        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')

        dist_maps_list = np.array(dist_maps_list)
        dist_maps_list = self.rescaleSegmentor.convert_to_segmentation(dist_maps_list)
        dist_maps_list = np.array(dist_maps_list)
        images_scores_list = np.array(images_scores_list)

        max_score = dist_maps_list.max()
        min_score = dist_maps_list.min()
        dist_maps_list = (dist_maps_list - min_score) / (max_score - min_score)
                                
        max_score = images_scores_list.max()
        min_score = images_scores_list.min()
        images_scores_list = (images_scores_list - min_score) / (max_score - min_score)

        image_dict = {
            "image":ori_imgs_list,
            "mask":mask_gt_list,
            "dist_map":dist_maps_list,
        }
        plot_images(self.out_path + "/test_images/", image_dict, denorm=True)
           
        image_test_results = metrics.compute_imagewise_retrieval_metrics(
            images_scores_list, images_labels_list
        )

        pix_test_results = metrics.compute_pixelwise_retrieval_metrics(
                dist_maps_list, mask_gt_list
        )

        # Compute PRO score & PW Auroc only images with anomalies
        sel_idxs = []
        for i in range(len(mask_gt_list)):
            if np.sum(mask_gt_list[i]) > 0:
                sel_idxs.append(i)

        anomaly_pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            [dist_maps_list[i] for i in sel_idxs],
            [mask_gt_list[i] for i in sel_idxs],
        )        
        
        logger.info('Test set image AUC: {:.2f}%'.format(100. * image_test_results["auroc"]))
        logger.info('Test set pix AUC: {:.2f}%'.format(100. * pix_test_results["auroc"]))
        logger.info('Test set anomaly pix AUC: {:.2f}%'.format(100. * anomaly_pixel_scores["auroc"]))
        self.results['test_time'] = test_time

        self.results['test_results'] = {
            'image_auroc':image_test_results['auroc'],
            'pixel_auroc':pix_test_results['auroc'],
            'pix_pro':anomaly_pixel_scores["auroc"]
        }

    
    
    