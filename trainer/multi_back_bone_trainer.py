from base.base_trainer import BaseTrainer

import logging
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
from utils import metrics
from utils.utils import plot_images, write_results
from networks.util import NetworkFeatureAggregator, RescaleSegmentor, PatchMaker
from networks.backbones import load_backbone
import torch.nn.functional as F
from networks.Feature_Contrastive import FeatureContrastiveBlock
from networks.util import get_gaussian_kernel
from networks.Fusion_Layer import Fusion_Layer

class MultiBackBoneTrainer(BaseTrainer):

    def __init__(self, trainer_para = {},  optim_para={}, scheduler_para={}, dataloader_para={}, network_init_para={}, out_path="", mode = "train", dataset_para = {}):
        
        trainer_para["extract_aug_view_layer"] = trainer_para["backbone_1"]["layers"]
        trainer_para["input_size_layer"] = trainer_para["backbone_1"]["layers"][0]

        self.backbone_aug = load_backbone(trainer_para["backbone_1"]["backbone_name"])
        self.feat_aug_agg = NetworkFeatureAggregator(self.backbone_aug, trainer_para["extract_aug_view_layer"], trainer_para["device"])
        self.feat_aug_agg.eval()            
        
        image_size = trainer_para["image_size"]
        rand_vec = torch.rand([1, 3, image_size, image_size]).to(trainer_para["device"])
        aug_out_put = self.feat_aug_agg(rand_vec)

        in_aug_ch = 0
        self.agg_aug_bi_bigger_size = aug_out_put[trainer_para["input_size_layer"]].shape[2]
       
        if "layer_size" in trainer_para and trainer_para["layer_size"] != 0:
            self.agg_aug_bi_bigger_size = trainer_para["layer_size"]
        
        for ilayer_name in trainer_para["extract_aug_view_layer"]:
            if trainer_para["backbone_1"]["backbone_name"] in trainer_para["backbone_1"]["tr_net"]:
                in_aug_ch = in_aug_ch + aug_out_put[ilayer_name].shape[-1]
            else:
                in_aug_ch = in_aug_ch + aug_out_put[ilayer_name].shape[1]
            
        backbone_1_ch = in_aug_ch
        backbone_2_ch = 0
        if "backbone_2" in trainer_para and trainer_para["backbone_2"]["mode"] == True:
            self.backbone_backbone_2 = load_backbone(trainer_para["backbone_2"]["backbone_name"])
            self.feat_backbone_2_agg = NetworkFeatureAggregator(self.backbone_backbone_2, trainer_para["backbone_2"]["layers"], trainer_para["device"])
            self.feat_backbone_2_agg.eval()
            aug_out_put = self.feat_backbone_2_agg(rand_vec)
            for ilayer_name in trainer_para["backbone_2"]["layers"]:
                backbone_2_curch = 0
                if trainer_para["backbone_2"]["backbone_name"] in trainer_para["backbone_2"]["tr_net"]:
                     backbone_2_curch = aug_out_put[ilayer_name].shape[-1]
                     in_aug_ch = in_aug_ch + backbone_2_curch
                else:
                    backbone_2_curch = aug_out_put[ilayer_name].shape[1]
                    in_aug_ch = in_aug_ch + backbone_2_curch
                backbone_2_ch = backbone_2_ch + backbone_2_curch
        self.fusion_layer = None
        if trainer_para["fusion_mbb"]["mode"] == True:
            if "fusion_size" in trainer_para["fusion_mbb"]:
                self.fusion_layer = Fusion_Layer(para=trainer_para["fusion_mbb"])
            else:
                self.fusion_layer = Fusion_Layer()

        self.rescaleSegmentor = RescaleSegmentor(trainer_para["device"], target_size=image_size, with_gaussion=False)
        self.blur_aug_layer = get_gaussian_kernel(kernel_size=3, sigma=4, channels=in_aug_ch).cuda()
        self.blur_aug_layer.eval()   

        if mode != "test":
            trainer_para["in_ch"] = in_aug_ch
            trainer_para["final_ch"] = in_aug_ch

            if trainer_para["score_method"] == "mse_patch":
                  trainer_para["in_ch"] = trainer_para["mse_patch_dim"]
                  trainer_para["final_ch"] = trainer_para["mse_patch_dim"]

            trainer_para["model_out_layers"] = trainer_para["aug_model_out_layers"]
            self.aug_model = self.set_network(trainer_para["net_name"], trainer_para, network_init_para)
            self.aug_model.to(trainer_para["device"])
            self.init_weights(self.aug_model)

        self.best_img_auc_roc = 0
        self.best_pix_auc_roc = 0

        self.dataset_para = dataset_para
        super().__init__(trainer_para, optim_para, scheduler_para, dataloader_para, network_init_para, out_path, mode)
        if mode != "test":
            self.optim_rec_aug = self.get_optimizer(self.aug_model, 
                                                        trainer_para["optimizer_name"],
                                                        optim_para["aug_view"]
                                                        )
            self.sched_aug = self.get_scheduler(self.optim_rec_aug,
                                                    trainer_para["scheduler_name"], 
                                                    scheduler_para)
           
        if "score_method" not in self.trainer_para:
            self.trainer_para["score_method"] = "patch_dis"

        if self.trainer_para["score_method"] in["patch_dis", "mse_patch", "patch_dis_mem", "mse_patch_dis"] :
            if "patch_reduce_dim" not in self.trainer_para:
                self.trainer_para["patch_reduce_dim"] = 384
            
            if "patch_score_size" not in self.trainer_para:
                self.trainer_para["patch_score_size"] = 3
            self.patch_make_score = PatchMaker(patchsize=self.trainer_para["patch_score_size"], dimension=self.trainer_para["patch_reduce_dim"])
    
    def resize_layer(self, cur_layer, back_bone_order = "backbone_1"):
        if self.trainer_para[back_bone_order]["backbone_name"] in self.trainer_para[back_bone_order]["tr_net"]:
            patch_shape = int(np.sqrt(cur_layer.shape[1]))
            dim = cur_layer.shape[-1]
            cur_layer = cur_layer.view(-1, patch_shape, patch_shape, dim)
            cur_layer = cur_layer.permute(0, 3, 1, 2)
            
        if cur_layer.shape[2] != self.agg_aug_bi_bigger_size:
            cur_layer = F.interpolate(
                cur_layer,
                size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                mode="bilinear",
                align_corners=False,
            )   
        return cur_layer
    
    def train(self, dataset:Dataset, test_dataset = None):
        # Get train data loader
        train_loader = DataLoader(dataset=dataset, shuffle=True, **self.dataloader_para)
        
        logger = logging.getLogger()
        # Training
        logger.info('Starting aug training...')
        logger.info('image size: '+ str(self.agg_aug_bi_bigger_size))

        if "fusion_size" in self.trainer_para["fusion_mbb"]:
            logger.info('fusion patch size: '+ str(self.trainer_para["fusion_mbb"]["fusion_size"]))

        start_time = time.time()

        self.aug_model.train()
        self.best_img_auc_roc = 0
        self.best_pix_auc_roc = 0
        for epoch in range(self.n_epochs):

            all_loss_epoch = 0.0
            loss_rec_epoch = 0.0
            loss_contrastive_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            with tqdm.tqdm(train_loader, desc="Train Model...", position=1, leave=False
                           ) as data_iterator:
                for inputs in data_iterator:
                    if isinstance(inputs, dict):
                        ori_img  = inputs["image"]
                        aug_img  = inputs["aug_image"]

                        #if self.dataset_para["cutpaste"]["mode"] == False:
                        #    aug_img = inputs["image"]
                        

                    ori_img = ori_img.to(self.device)
                    aug_img = aug_img.to(self.device)

                    # extract_feature
                    with torch.no_grad():
                        out_agg_ori = self.feat_aug_agg(ori_img)         
                        out_agg_aug = self.feat_aug_agg(aug_img)  

                        if "backbone_2" in self.trainer_para and self.trainer_para["backbone_2"]["mode"] == True:
                            out_agg_backbone_2_ori = self.feat_backbone_2_agg(ori_img)         
                            out_agg_backbone_2_aug = self.feat_backbone_2_agg(aug_img)  

                    backbone_1_aug_layer_list = []
                    backbone_1_ori_layer_list = []

                    backbone_2_aug_layer_list = []
                    backbone_2_ori_layer_list = []
                    with torch.no_grad():
                        for ilayer_name in self.trainer_para["extract_aug_view_layer"]:
                            aug_layer = out_agg_aug[ilayer_name]
                            ori_layer = out_agg_ori[ilayer_name]

                            aug_layer = self.resize_layer(aug_layer, "backbone_1")
                            ori_layer = self.resize_layer(ori_layer, "backbone_1")
                            
                            backbone_1_aug_layer_list.append(aug_layer)
                            backbone_1_ori_layer_list.append(ori_layer)
                        
                        loss_con = 0
                        if "backbone_2" in self.trainer_para and self.trainer_para["backbone_2"]["mode"] == True:

                            for ilayer_name in self.trainer_para["backbone_2"]["layers"]:
                            
                                aug_layer = out_agg_backbone_2_aug[ilayer_name]
                                ori_layer = out_agg_backbone_2_ori[ilayer_name]

                                aug_layer = self.resize_layer(aug_layer, "backbone_2")
                                ori_layer = self.resize_layer(ori_layer, "backbone_2")
                                backbone_2_aug_layer_list.append(aug_layer)
                                backbone_2_ori_layer_list.append(ori_layer)  

                            
                    if self.trainer_para["backbone_2"]["mode"] == True:
                        ori_input = None
                        aug_input = None
                        
               
                        if "concat" in self.trainer_para["backbone_2"] and self.trainer_para["backbone_2"]["concat"] == "oc":
                            ori_input_list = []
                            aug_input_list = []

                            b1_ind = 0
                            b2_ind = 0
                            for i_l_name in self.trainer_para["backbone_2"]["concat"]["order"]:
                                if i_l_name == "b1":
                                    ori_input_list.append(backbone_1_ori_layer_list[b1_ind])
                                    aug_input_list.append(backbone_1_aug_layer_list[b1_ind])
                                    b1_ind += 1
                                
                                elif i_l_name == "b2":
                                    ori_input_list.append(backbone_2_ori_layer_list[b2_ind])
                                    aug_input_list.append(backbone_2_aug_layer_list[b2_ind])
                                    b2_ind += 1
                                    
                            ori_input = torch.cat(ori_input_list, 1) 
                            aug_input = torch.cat(aug_input_list, 1)
                        else:
                            backbone_1_aug_feature = torch.cat(backbone_1_aug_layer_list, 1) 
                            backbone_1_ori_feature = torch.cat(backbone_1_ori_layer_list, 1)

                            backbone_2_aug_feature = torch.cat(backbone_2_aug_layer_list, 1)
                            backbone_2_ori_feature = torch.cat(backbone_2_ori_layer_list, 1)

                            ori_input = torch.cat([backbone_1_ori_feature, backbone_2_ori_feature], 1)    
                            aug_input = torch.cat([backbone_1_aug_feature, backbone_2_aug_feature], 1)
                    else:

                        backbone_1_aug_feature = torch.cat(backbone_1_aug_layer_list, 1) 
                        backbone_1_ori_feature = torch.cat(backbone_1_ori_layer_list, 1)
                    
                        ori_input = backbone_1_ori_feature
                        aug_input = backbone_1_aug_feature
                    
                    if self.trainer_para["blur"]["aug"] == True:
                        aug_input = self.blur_aug_layer(aug_input)
                    
                    if self.trainer_para["blur"]["aug"] == True:
                        ori_input = self.blur_aug_layer(ori_input)

                    self.optim_rec_aug.zero_grad()  

                    out_aug = self.aug_model(aug_input)


                    if self.trainer_para["fusion_mbb"]["mode"] == True:
                        loss_con = self.fusion_layer(ori_input, out_aug)
                       
                    loss_mse = self.criterion(ori_input, out_aug)
                    con_lam = self.trainer_para["loss_contrastive_beta"]
                    all_loss = loss_mse + con_lam * loss_con
                    #print(con_lam)
                    all_loss.backward()
                    self.optim_rec_aug.step()


                    all_loss_epoch += all_loss.item()
                    loss_rec_epoch += loss_mse.item()
                    loss_contrastive_epoch += loss_con
                    n_batches += 1

            if self.scheduler_name == "plateau":    
                self.sched_aug.step(loss_rec_epoch)
            else:
                self.sched_aug.step()
                
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}  Time: {:.3f} All_Loss:{:.5f}  Loss_Rec: {:.5f}  Loss_cont: {:.5f}'
                        .format(epoch + 1, 
                                self.n_epochs, 
                                epoch_train_time, 
                                all_loss_epoch / n_batches,
                                loss_rec_epoch / n_batches,
                                loss_contrastive_epoch / n_batches
                                ))
            
            if self.trainer_para["logger"]["epoch_test"] == True:
                self.test(self.aug_model, test_dataset)
                if "epoch_results" not in self.results:
                    self.results["epoch_results"] = []
                self.results['test_results']["epoch"] = epoch
                self.results['test_results']["loss_rec_epoch"] = float(loss_rec_epoch)/n_batches
                self.results['test_results']["all_loss_epoch"] = float(all_loss_epoch)/n_batches
                self.results['test_results']["loss_contrastive_epoch"] = float(loss_contrastive_epoch)/n_batches
                self.results["epoch_results"].append(self.results['test_results'])

            if self.results['test_results']["image_auroc"] > self.best_img_auc_roc:
                self.best_img_auc_roc = self.results['test_results']["image_auroc"]
                self.best_pix_auc_roc = self.results['test_results']["pixel_auroc"]
                self.save_network(self.aug_model, "best_model")
                self.results["best_epoch"] = epoch
                self.results["best_epoch_result"] = self.results['test_results']
                
            elif self.results['test_results']["image_auroc"] == self.best_img_auc_roc:
                if self.results['test_results']["pixel_auroc"] > self.best_pix_auc_roc:
                    self.best_pix_auc_roc = self.results['test_results']["pixel_auroc"]
                    self.save_network(self.aug_model, "best_model")
                    self.results["best_epoch"] = epoch
                    self.results["best_epoch_result"] = self.results['test_results']

            self.save_results()    
            if (epoch) % 5 == 0:
                with torch.no_grad():
                    array_aug_input  = aug_input[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_aug_output = out_aug[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_aug_img    = aug_img[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    
                    
                    array_ori_input = ori_input[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_out_ori = array_ori_input

                    remaps_ori = (array_aug_input - array_aug_output)**2
                    dis_maps = np.sqrt(np.mean(remaps_ori, axis=3))
                   
                    images_dict = {
                        "aug_img":array_aug_img,
                        "aug_input":np.sum(array_aug_input, axis=3),
                        "aug_output":np.sum(array_aug_output, axis=3),
                        "ori_input":np.sum(array_ori_input, axis=3),
                        "out_ori":np.sum(array_out_ori, axis=3),
                        "dis_maps":dis_maps
                    }
                    plot_images(self.out_path + "/train_aug_images/", images_dict, denorm=True)

            

            if (epoch + 1) % 5 == 0:
                #if self.trainer_para["fusion_mbb"]["mode"] == True:
                #    self.save_network(self.fusion_layer, "fusion_layer")

                self.save_network(self.aug_model, "aug_model")

        train_time = time.time() - start_time
        logger.info('Train time: %.3f' % train_time)
        logger.info('Finished aug training.')
        self.results["train_time"] = train_time

    def test(self, model, dataset):
        logger = logging.getLogger()

        # Get test data loader
        test_loader    = DataLoader(dataset=dataset, shuffle=False, **self.dataloader_para)

        # Testing
        logger.info('Testing autoencoder...')
        start_time = time.time()

        mask_gt_list = []
        dist_maps_list = []
        ori_imgs_list = []
        images_scores_list = []
        images_labels_list = []

        feat_agg = self.feat_aug_agg
        model.eval()


        with torch.no_grad():
            with tqdm.tqdm(test_loader, desc="Val Model...", position=1, leave=False
                           ) as data_iterator:
                
                for data in data_iterator:
                    input_imgs, masks, labels = data["image"], data["mask"], data['label']
                    anomaly =  data['anomaly']

                    input_imgs = input_imgs.to(self.device)

                    img_agg = feat_agg(input_imgs)
                    
                    backbone_1_img_agg_layer_list = []
                    backbone_2_img_agg_layer_list = []
                    img_agg_input = None
                    for ilayer_name in self.trainer_para["extract_aug_view_layer"]:
                        img_layer = img_agg[ilayer_name]
                        img_layer = self.resize_layer(img_layer, "backbone_1")
                        backbone_1_img_agg_layer_list.append(img_layer)

                    if "backbone_2" in self.trainer_para and self.trainer_para["backbone_2"]["mode"] == True:
                        backbone_2_agg = self.feat_backbone_2_agg
                        backbone_2_img = backbone_2_agg(input_imgs)
                        for ilayer_name in self.trainer_para["backbone_2"]["layers"]:
                            img_layer = backbone_2_img[ilayer_name]

                            img_layer = self.resize_layer(img_layer, "backbone_2")
                            backbone_2_img_agg_layer_list.append(img_layer)

                        if "concat" in self.trainer_para["backbone_2"] and self.trainer_para["backbone_2"]["concat"]["mode"] == "oc":
                            img_agg_input_list = []
                            b1_ind = 0
                            b2_ind = 0
                            for i_l_name in self.trainer_para["backbone_2"]["concat"]["order"]:
                                if i_l_name == "b1":
                                    img_agg_input_list.append(backbone_1_img_agg_layer_list[b1_ind])
                                    b1_ind += 1
                                elif i_l_name == "b2":
                                    img_agg_input_list.append(backbone_2_img_agg_layer_list[b2_ind])
                                    b2_ind += 1
                            img_agg_input = torch.cat(img_agg_input_list, 1)
                        
                        else:
                            backbone_1_fea = torch.cat(backbone_1_img_agg_layer_list, 1)
                            backbone_2_fea = torch.cat(backbone_2_img_agg_layer_list, 1)
                            img_agg_input = torch.cat([backbone_1_fea, backbone_2_fea], 1)
                    else:

                        backbone_1_fea = torch.cat(backbone_1_img_agg_layer_list, 1)
                        img_agg_input = backbone_1_fea

                    if self.trainer_para["blur"]["aug"] == True:
                        img_agg_input = self.blur_aug_layer(img_agg_input)

                    if self.trainer_para["score_method"] == "mse_patch":
                        img_agg_input, patch_shape = self.patch_make_score._embed(img_agg_input, detech_fea = False, provide_patch_shapes=True)
                        batch_size = img_agg_input.shape[0]
                        img_agg_input = img_agg_input.view (batch_size, patch_shape[0], patch_shape[1], -1)
                        img_agg_input = img_agg_input.permute(0, 3, 1, 2)

                    if len(self.trainer_para["model_out_layers"]) != 0:
                        outputs, _ = model(img_agg_input)
                    else:
                        outputs = model(img_agg_input)

                    if self.trainer_para["score_method"] == "mse":
                        array_inputs  = img_agg_input.cpu().numpy()
                        array_outputs = outputs.cpu().numpy()
                        remaps_ori = (array_inputs - array_outputs)**2
                        remaps = np.sqrt(np.mean(remaps_ori, axis=1))
                        #scores = np.max(remaps, axis=(1, 2))
                        scores = np.sqrt(np.mean(np.max(remaps_ori, axis=(2, 3)), axis=1))

                    elif self.trainer_para["score_method"] == "cos_sim":
                       remaps = 1 - F.cosine_similarity(img_agg_input, outputs).cpu().numpy()
                       scores = np.max(remaps, axis=(1, 2))

                    elif self.trainer_para["score_method"] == "mse_patch":
                        array_inputs  = img_agg_input.cpu().numpy()
                        array_outputs = outputs.cpu().numpy()

                        abs_fea_score = np.sqrt((array_inputs - array_outputs)**2)
                        scores = np.mean(np.max(abs_fea_score, axis=(-1, -2)), axis = -1)

                        fea_score = np.mean(abs_fea_score, axis=1)
                        remaps = fea_score.reshape(fea_score.shape[0], patch_shape[0], patch_shape[1])

                    elif self.trainer_para["score_method"] == "mse_patch_dis":
                        array_inputs  = img_agg_input.cpu().numpy()
                        array_outputs = outputs.cpu().numpy()

                        abs_fea_score = np.sqrt((array_inputs - array_outputs)**2)
                        mse_scores = np.mean(np.max(abs_fea_score, axis=(-1, -2)), axis = -1)

                        fea_out, patch_shape = self.patch_make_score._embed(outputs, provide_patch_shapes=True)
                        fea_in = self.patch_make_score._embed(img_agg_input)
                        
                        abs_fea_score = np.sqrt((fea_out - fea_in)**2)
                        
                        #instance score
                        max_mean_score = np.mean(np.max(abs_fea_score, axis=(1)), axis = -1)
                        mean_max_score = np.max(np.mean(abs_fea_score, axis=(-1)), axis = -1)

                        if "patch_score_lr" not in self.trainer_para:
                            lr = 0.6
                        else:
                            lr = self.trainer_para["patch_score_lr"]

                        scores = mse_scores + lr * max_mean_score + mean_max_score * (1 - lr)

                        #pix score
                        fea_score = np.mean(abs_fea_score, axis=-1)
                        remaps = fea_score.reshape(fea_score.shape[0], patch_shape[0], patch_shape[1])

                    elif self.trainer_para["score_method"] == "patch_dis":
                       
                        fea_out, patch_shape = self.patch_make_score._embed(outputs, provide_patch_shapes=True)
                        fea_in = self.patch_make_score._embed(img_agg_input)
                        abs_fea_score = np.sqrt((fea_out - fea_in)**2)
                      
                        max_mean_score = np.mean(np.max(abs_fea_score, axis=(1)), axis = -1)
                        mean_max_score = np.max(np.mean(abs_fea_score, axis=(-1)), axis = -1)

                        if "patch_score_lr" not in self.trainer_para:
                            lr = 1
                        else:
                            lr = self.trainer_para["patch_score_lr"]
                        #print(lr)

                        scores = lr * max_mean_score + mean_max_score * (1 - lr) 

                        #pix score
                        fea_score = np.mean(abs_fea_score, axis=-1)
                        remaps = fea_score.reshape(fea_score.shape[0], patch_shape[0], patch_shape[1])
                             
                    # transpose for display
                    mask_gt_list.extend([i_mask.squeeze() for i_mask in masks.cpu().numpy()])
                    dist_maps_list.extend([iMap  for iMap in remaps])
                    ori_imgs_list.extend([i_ori_img for i_ori_img in input_imgs.cpu().numpy().transpose(0, 2, 3, 1)])
                    images_scores_list.extend(scores)
                    images_labels_list.extend(labels.tolist())
        
        

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

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % (test_time / len(images_scores_list)))
        logger.info('Finished testing autoencoder.')
        
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
        self.results['test_time'] = test_time / images_scores_list.shape[0]

        self.results['test_results'] = {
            'image_auroc':image_test_results['auroc']*100.,
            'pixel_auroc':pix_test_results['auroc']*100.,
            'pix_pro':anomaly_pixel_scores["auroc"]*100.,
            'patch_score_lr':self.trainer_para["patch_score_lr"],
            'patch_score_size':self.trainer_para["patch_score_size"],
            'patch_reduce_dim':self.trainer_para["patch_reduce_dim"]
        }

        if self.mode != "test":
            out_path = "output/train/"
        else:
            out_path = "output/test/"
        
        if self.results['test_results']["image_auroc"] > self.best_img_auc_roc:
            write_results(
                self.results['test_results'],
                self.trainer_para["class_name"],
                self.trainer_para["class_list"],
                out_path +"/"+ self.trainer_para["dataset_name"] + "/" + self.trainer_para["outdir_final_index_name"]+ ".csv"
            )
        elif self.results['test_results']["image_auroc"] == self.best_img_auc_roc:
            if self.results['test_results']["pixel_auroc"] > self.best_pix_auc_roc:
                write_results(
                    self.results['test_results'],
                    self.trainer_para["class_name"],
                    self.trainer_para["class_list"],
                    out_path +"/"+ self.trainer_para["dataset_name"] + "/" + self.trainer_para["outdir_final_index_name"]+ ".csv"
                )


        if self.trainer_para["save_test_image"] == True:
            image_dict = {
                "image":ori_imgs_list,
                "mask":mask_gt_list,
                "dist_map":dist_maps_list,
            }
            plot_images(self.out_path + "/test_images/", image_dict, denorm=True)

    
    
    