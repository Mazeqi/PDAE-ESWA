from base.base_trainer import BaseTrainer

import logging
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
from utils import metrics
from utils.utils import plot_images, write_results
from networks.util import NetworkFeatureAggregator, RescaleSegmentor, PatchMaker, AggregatorBackbone
from networks.backbones import load_backbone
import torch.nn.functional as F
from networks.Feature_Contrastive import FeatureContrastiveBlock
from networks.util import get_gaussian_kernel
import faiss

class FLOW_AE_BASE_Trainer(BaseTrainer):

    def __init__(self, trainer_para = {},  optim_para={}, scheduler_para={}, dataloader_para={}, network_init_para={}, out_path="", mode = "train", dataset_para = {}):
        
        if trainer_para["dataset_name"] not in trainer_para["network_layer"][trainer_para["aug_backbone_name"]]:
            trainer_para["extract_aug_view_layer"] = trainer_para["network_layer"][trainer_para["aug_backbone_name"]]["default"]["layers"]
            trainer_para["input_size_layer"] = trainer_para["network_layer"][trainer_para["aug_backbone_name"]]["default"]["input_size_layer"]
            trainer_para["extract_ori_view_layer"] = trainer_para["extract_aug_view_layer"]
            
        else:
            trainer_para["extract_aug_view_layer"] = trainer_para["network_layer"][trainer_para["aug_backbone_name"]][trainer_para["dataset_name"]][trainer_para["class_name"]]["layers"]
            trainer_para["input_size_layer"] = trainer_para["network_layer"][trainer_para["aug_backbone_name"]][trainer_para["dataset_name"]][trainer_para["class_name"]]["input_size_layer"]
            trainer_para["extract_ori_view_layer"] = trainer_para["extract_aug_view_layer"]

        if mode == "mv_train":
            self.backbone_ori = load_backbone(trainer_para["ori_backbone_name"])
            self.feat_ori_agg = NetworkFeatureAggregator(self.backbone_ori,  trainer_para["extract_ori_view_layer"], trainer_para["device"])
            self.feat_ori_agg.eval()

        self.backbone_aug = load_backbone(trainer_para["aug_backbone_name"])
        self.feat_aug_agg = NetworkFeatureAggregator(self.backbone_aug, trainer_para["extract_aug_view_layer"], trainer_para["device"])
        self.feat_aug_agg.eval()

        if "cait" in trainer_para and trainer_para["cait"]["mode"] == True:
            self.backbone_cait = load_backbone(trainer_para["cait"]["backbone_name"])
            self.feat_cait_agg = NetworkFeatureAggregator(self.backbone_cait, trainer_para["cait"]["layers"], trainer_para["device"])
            self.feat_cait_agg.eval()
        
        image_size = trainer_para["image_size"]
        if "crop_size" in dataset_para:
            image_size = dataset_para["crop_size"]

        rand_vec = torch.rand([1, 3, image_size, image_size]).to(trainer_para["device"])

        aug_out_put = self.feat_aug_agg(rand_vec)

        if mode == "mv_train":
            ori_out_put = self.feat_ori_agg(rand_vec)

        in_aug_ch = 0
        self.agg_aug_bi_bigger_size = aug_out_put[trainer_para["input_size_layer"]].shape[2]
       
        if "layer_size" in trainer_para and trainer_para["layer_size"] != 0:
            self.agg_aug_bi_bigger_size = trainer_para["layer_size"]
        
        #if "cait" in trainer_para and trainer_para["cait"]["mode"] == True:
        #    self.agg_aug_bi_bigger_size = 96
        
        for ilayer_name in trainer_para["extract_aug_view_layer"]:
            in_aug_ch = in_aug_ch + aug_out_put[ilayer_name].shape[1]
            
           
        if mode == "mv_train":
            self.agg_ori_bi_bigger_size = self.agg_aug_bi_bigger_size
            in_ori_ch = 0
            for ilayer_name in trainer_para["extract_ori_view_layer"]:
                in_ori_ch = in_ori_ch + ori_out_put[ilayer_name].shape[1]
        
        if "cait" in trainer_para and trainer_para["cait"]["mode"] == True:

            aug_out_put = self.feat_cait_agg(rand_vec)

            for ilayer_name in trainer_para["cait"]["layers"]:
                if trainer_para["cait"]["backbone_name"] in trainer_para["cait"]["tr_net"]:
                     in_aug_ch = in_aug_ch + aug_out_put[ilayer_name].shape[-1]
                else:
                    in_aug_ch = in_aug_ch + aug_out_put[ilayer_name].shape[1]
            
            if mode == "mv_train":
                in_ori_ch = in_aug_ch
        if "layer_agg" in trainer_para and trainer_para["layer_agg"]["mode"] == True:
            in_aug_ch = trainer_para["layer_agg"]["layer_ch"]
            in_ori_ch = trainer_para["layer_agg"]["layer_ch"]
            self.layer_agg = AggregatorBackbone(in_aug_ch)
            self.layer_agg.eval()   

        self.rescaleSegmentor = RescaleSegmentor(trainer_para["device"], target_size=image_size, with_gaussion=False)
        self.blur_aug_layer = get_gaussian_kernel(kernel_size=3, sigma=4, channels=in_aug_ch).cuda()
        self.blur_aug_layer.eval()   

        if mode == "mv_train":
            self.blur_ori_layer = get_gaussian_kernel(kernel_size=3, sigma=4, channels=in_ori_ch).cuda()
            self.blur_ori_layer.eval()     

            trainer_para["in_ch"] = in_ori_ch
            trainer_para["final_ch"] = in_ori_ch
            trainer_para["model_out_layers"] = trainer_para["ori_model_out_layers"]

            if trainer_para["score_method"] == "mse_patch":
                  trainer_para["in_ch"] = trainer_para["mse_patch_dim"]
                  trainer_para["final_ch"] = trainer_para["mse_patch_dim"]

            self.ori_model = self.set_network(trainer_para["net_ori_name"], trainer_para, network_init_para)
            self.ori_model.to(trainer_para["device"])
            self.init_weights(self.ori_model)

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
        
        if mode == "mv_train":
            self.fea_contrastive = FeatureContrastiveBlock()
            self.fea_contrastive.to(trainer_para["device"])

        

        super().__init__(trainer_para, optim_para, scheduler_para, dataloader_para, network_init_para, out_path, mode)
        if mode != "test":
            self.optim_rec_aug = self.get_optimizer(self.aug_model, 
                                                    trainer_para["optimizer_name"],
                                                    optim_para["aug_view"]
                                                    )
            self.sched_aug = self.get_scheduler(self.optim_rec_aug,
                                                    trainer_para["scheduler_name"], 
                                                    scheduler_para)
        if mode == "mv_train":
            self.optim_rec_ori = self.get_optimizer(self.ori_model, 
                                                    trainer_para["optimizer_name"],
                                                    optim_para["ori_view"]
                                                    )
            
            self.sched_ori = self.get_scheduler(self.optim_rec_ori,
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

    def train(self, dataset:Dataset):
        # Get train data loader
        train_loader = DataLoader(dataset=dataset, shuffle=True, **self.dataloader_para)
        
        logger = logging.getLogger()
        # Training
        logger.info('Starting aug training...')
        logger.info('image size: '+ str(self.agg_aug_bi_bigger_size))
        start_time = time.time()

        self.aug_model.train()

        if self.mode == "mv_train":
            self.ori_model.train()

        for epoch in range(self.n_epochs):

            all_loss_epoch = 0.0
            loss_aug_epoch = 0.0
            loss_contrastive_epoch = 0.0
            loss_ori_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            with tqdm.tqdm(train_loader, desc="Train Model...", position=1, leave=False
                           ) as data_iterator:
                for inputs in data_iterator:
                    if isinstance(inputs, dict):
                        ori_img  = inputs["image"]
                        aug_img  = inputs["aug_image"]

                    ori_img = ori_img.to(self.device)
                    aug_img = aug_img.to(self.device)

                    with torch.no_grad():
                        # ori view
                        if self.mode == "mv_train":
                            out_agg_ori = self.feat_ori_agg(ori_img)

                        # aug view
                        out_aug_agg_ori = self.feat_aug_agg(ori_img)         
                        out_agg_aug     = self.feat_aug_agg(aug_img)  


                        if "cait" in self.trainer_para and self.trainer_para["cait"]["mode"] == True:
                            out_aug_agg_cait_ori = self.feat_cait_agg(ori_img)         
                            out_agg_aug_cait     = self.feat_cait_agg(aug_img)  

                            #if self.mode == "mv_train":
                            #    out_ori_agg_cait = self.feat_cait_agg(ori_img)

                    ori_layer_list = []
                    aug_layer_list = []
                    aug_ori_layer_list = []
                    with torch.no_grad():
                        if self.mode == "mv_train":
                            for ilayer_name in self.trainer_para["extract_ori_view_layer"]:
                                ori_layer = out_agg_ori[ilayer_name]

                                if ori_layer.shape[2] != self.agg_ori_bi_bigger_size:
                                    ori_layer = F.interpolate(
                                        ori_layer,
                                        size=(self.agg_ori_bi_bigger_size, self.agg_ori_bi_bigger_size),
                                        mode="bilinear",
                                        align_corners=False,
                                    ) 
                                  
                                ori_layer_list.append(ori_layer)


                        for ilayer_name in self.trainer_para["extract_aug_view_layer"]:
                            
                            aug_layer     = out_agg_aug[ilayer_name]
                            aug_ori_layer = out_aug_agg_ori[ilayer_name]
                        
                            if aug_layer.shape[2] != self.agg_aug_bi_bigger_size:
                                aug_layer = F.interpolate(
                                    aug_layer,
                                    size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                    mode="bilinear",
                                    align_corners=False,
                                ) 
                              
                            if aug_ori_layer.shape[2] != self.agg_aug_bi_bigger_size:
                                aug_ori_layer = F.interpolate(
                                    aug_ori_layer,
                                    size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                    mode="bilinear",
                                    align_corners=False,
                                )       
                            #print(aug_layer.shape)
                            aug_layer_list.append(aug_layer)
                            aug_ori_layer_list.append(aug_ori_layer)

                        if "cait" in self.trainer_para and self.trainer_para["cait"]["mode"] == True:
                            #if self.mode == "mv_train":
                            #    for ilayer_name in self.trainer_para["cait"]["layers"]:

                            for ilayer_name in self.trainer_para["cait"]["layers"]:
                            
                                aug_layer     = out_agg_aug_cait[ilayer_name]
                                aug_ori_layer = out_aug_agg_cait_ori[ilayer_name]
                                
                                if self.trainer_para["cait"]["backbone_name"] in self.trainer_para["cait"]["tr_net"]:
                                    patch_shape = int(np.sqrt(aug_layer.shape[1]))
                                    aug_layer = aug_layer.view(-1, patch_shape, patch_shape, 192)
                                    aug_ori_layer = aug_ori_layer.view(-1, patch_shape, patch_shape, 192)

                                    aug_layer = aug_layer.permute(0, 3, 1, 2)
                                    aug_ori_layer = aug_ori_layer.permute(0, 3, 1, 2)
                                

                                if aug_layer.shape[2] != self.agg_aug_bi_bigger_size:
                                    aug_layer = F.interpolate(
                                        aug_layer,
                                        size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                        mode="bilinear",
                                        align_corners=False,
                                    ) 
                                
                                if aug_ori_layer.shape[2] != self.agg_aug_bi_bigger_size:
                                    aug_ori_layer = F.interpolate(
                                        aug_ori_layer,
                                        size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                        mode="bilinear",
                                        align_corners=False,
                                    )       
                                aug_layer_list.append(aug_layer)
                                aug_ori_layer_list.append(aug_ori_layer)

                                if self.mode == "mv_train":
                                    ori_layer_list.append(aug_ori_layer)
                        
                        if self.mode == "mv_train":
                            
                            ori_input = torch.cat(ori_layer_list, 1)

                            if "layer_agg" in self.trainer_para and self.trainer_para["layer_agg"]["mode"] == True:
                                batch_size = ori_input.shape[0]
                                dim_len = ori_input.shape[1]
                                patch_shape = ori_input.shape[2]
                                
                                ori_input = ori_input.permute(0, 2, 3, 1).reshape(-1, dim_len)
                                ori_input = self.layer_agg(ori_input).reshape(batch_size, patch_shape, patch_shape, -1)
                                ori_input = ori_input.permute(0, 3, 1, 2)

                            ori_input = self.blur_ori_layer(ori_input)
                        
                        aug_input = torch.cat(aug_layer_list, 1) 
                        aug_ori_input = torch.cat(aug_ori_layer_list, 1)    

                        if "layer_agg" in self.trainer_para and self.trainer_para["layer_agg"]["mode"] == True:                         
                            batch_size = aug_input.shape[0]
                            dim_len = aug_input.shape[1]
                            patch_shape = aug_input.shape[2]

                            aug_input = aug_input.permute(0, 2, 3, 1).reshape(-1, dim_len)
                            aug_input = self.layer_agg(aug_input).reshape(batch_size, patch_shape, patch_shape, -1)
                            aug_input = aug_input.permute(0, 3, 1, 2)

                            aug_ori_input = aug_ori_input.permute(0, 2, 3, 1).reshape(-1, dim_len)
                            aug_ori_input = self.layer_agg(aug_ori_input).reshape(batch_size, patch_shape, patch_shape, -1)
                            aug_ori_input = aug_ori_input.permute(0, 3, 1, 2)

                        #print(aug_input.shape)
                        aug_input = self.blur_aug_layer(aug_input)
                        aug_ori_input = self.blur_aug_layer(aug_ori_input)

                        #patch recover
                        if self.trainer_para["score_method"] == "mse_patch":
                            aug_input, patch_shape = self.patch_make_score._embed(aug_input, detech_fea = False, provide_patch_shapes=True)
                            aug_ori_input = self.patch_make_score._embed(aug_ori_input, detech_fea = False, provide_patch_shapes=False)
                            
                            batch_size = aug_input.shape[0]
                            aug_input = aug_input.view (batch_size, patch_shape[0], patch_shape[1], -1)
                            aug_input = aug_input.permute(0, 3, 1, 2)
                            aug_ori_input = aug_ori_input.view (batch_size, patch_shape[0], patch_shape[1], -1)
                            aug_ori_input = aug_ori_input.permute(0, 3, 1, 2)

                            if self.mode == "mv_train":
                                ori_input = self.patch_make_score._embed(ori_input, detech_fea = False, provide_patch_shapes=False)
                                ori_input = ori_input.view (batch_size, patch_shape[0], patch_shape[1], -1)
                                ori_input = ori_input.permute(0, 3, 1, 2)
                    # Zero the network parameter gradients
                    self.optim_rec_aug.zero_grad()

                    if self.mode == "mv_train":
                        self.optim_rec_ori.zero_grad()
                
                    # Update network parameters via backpropagation: forward + backward + optimize
                    if len(self.trainer_para["model_out_layers"]) != 0:
                        out_aug, aug_middle_layers = self.aug_model(aug_input)

                        if self.mode == "mv_train":
                            out_ori, ori_middle_layers = self.ori_model(ori_input)
                    else:
                        out_aug = self.aug_model(aug_input)
                        
                        if self.mode == "mv_train":
                            out_ori = self.ori_model(ori_input)

                    if self.loss_name == "cos_sim":
                        loss_aug = torch.mean(1 - self.criterion(aug_ori_input.view(aug_ori_input.shape[0], -1), out_aug.view(out_aug.shape[0], -1)))
                    elif self.loss_name == "mse":
                        loss_aug = self.criterion(aug_ori_input, out_aug)
                    
                    if self.mode == "mv_train":
                        if self.loss_name == "cos_sim":
                            loss_ori = torch.mean(1 - self.criterion(ori_input.view(ori_input.shape[0], -1), out_ori.view(out_ori.shape[0], -1)))
                        elif self.loss_name == "mse":
                            loss_ori = self.criterion(ori_input, out_ori)

                        loss_cont = self.fea_contrastive(aug_middle_layers, ori_middle_layers, self.trainer_para["contrastive_loss"])
                    else:
                        loss_cont = 0
                        loss_ori = 0

                    if self.mode == "mv_train":         
                        loss_contrastive_beta = self.trainer_para["loss_contrastive_beta"]

                        #if (epoch+1) % 150 == 0:
                        #    loss_contrastive_beta *= 10
                        all_loss = loss_aug  + loss_ori + loss_contrastive_beta * loss_cont
                    else:
                        all_loss = loss_aug

                    all_loss.backward()
                    self.optim_rec_aug.step()

                    if self.mode == "mv_train":
                        self.optim_rec_ori.step()

                    all_loss_epoch += all_loss.item()
                    loss_aug_epoch += loss_aug.item()

                    if self.mode == "mv_train": 
                        loss_ori_epoch += loss_ori.item()
                        loss_contrastive_epoch += loss_cont
                    else:
                        loss_ori_epoch += 0
                        loss_contrastive_epoch += 0
                    n_batches += 1

            if self.scheduler_name == "plateau":    
                self.sched_aug.step(loss_aug_epoch)

                if self.mode == "mv_train": 
                    self.sched_ori.step(loss_ori_epoch)
            else:
                self.sched_aug.step()
                if self.mode == "mv_train": 
                    self.sched_ori.step()
                
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}  Time: {:.3f} All_Loss:{:.5f}  Loss_Aug: {:.5f}  Loss_Ori: {:.5f} Loss_cont: {:.5f}'
                        .format(epoch + 1, 
                                self.n_epochs, 
                                epoch_train_time, 
                                all_loss_epoch / n_batches,
                                loss_aug_epoch / n_batches,
                                loss_ori_epoch / n_batches,
                                loss_contrastive_epoch / n_batches
                                ))

            if (epoch) % 5 == 0:
                with torch.no_grad():
                    array_aug_input  = aug_input[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_aug_output = out_aug[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    array_aug_img    = aug_img[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    
                    if self.mode == "mv_train":
                        array_ori_input  = ori_input[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                        array_out_ori = out_ori[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
                    else:
                        array_ori_input = aug_ori_input[:1,:, :, :].cpu().numpy().transpose(0, 2, 3, 1)
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
                
                if self.mode == "mv_train":
                    self.save_network(self.ori_model, "ori_model")

                self.save_network(self.aug_model, "aug_model")
        
        train_time = time.time() - start_time
        logger.info('Train time: %.3f' % train_time)
        logger.info('Finished aug training.')
        self.results["train_time"] = train_time

    def memory_bank(self, memory_dataset):
        memory_loader = DataLoader(dataset=memory_dataset, shuffle=True, batch_size=8)
        feat_agg = self.feat_aug_agg
        fea_mem = None
        with torch.no_grad():
            with tqdm.tqdm(memory_loader, desc="Memory Bank...", position=1, leave=False) as data_iterator:
                for data in data_iterator:
                    input_imgs = data["image"]
                    input_imgs = input_imgs.to(self.device)

                    img_agg = feat_agg(input_imgs)
                    
                    img_agg_layer_list = []
                    for ilayer_name in self.trainer_para["extract_aug_view_layer"]:
                        img_layer = img_agg[ilayer_name]
                        if img_layer.shape[2] != self.agg_aug_bi_bigger_size:
                            img_layer = F.interpolate(
                                img_layer,
                                size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                mode="bilinear",
                                align_corners=False,
                            ) 
                        img_agg_layer_list.append(img_layer)
                    img_agg_input = torch.cat(img_agg_layer_list, 1)
                    img_agg_input = self.blur_aug_layer(img_agg_input)
                    fea_mem = self.patch_make_score._embed(img_agg_input, detech_fea=False)
                    break

        return fea_mem

    def test(self, model, dataset, memory_dataset = None):
        logger = logging.getLogger()

        # Get test data loader
        test_loader    = DataLoader(dataset=dataset, shuffle=False, **self.dataloader_para)

        if self.trainer_para["score_method"] == "patch_dis_mem":
            fea_memory = self.memory_bank(memory_dataset)

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
                    input_imgs = input_imgs.to(self.device)

                    img_agg = feat_agg(input_imgs)
                    
                    img_agg_layer_list = []
                    for ilayer_name in self.trainer_para["extract_aug_view_layer"]:
                        img_layer = img_agg[ilayer_name]
                        if img_layer.shape[2] != self.agg_aug_bi_bigger_size:
                            img_layer = F.interpolate(
                                img_layer,
                                size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                mode="bilinear",
                                align_corners=False,
                            ) 
                        img_agg_layer_list.append(img_layer)
                    

                    if "cait" in self.trainer_para and self.trainer_para["cait"]["mode"] == True:
                        cait_agg = self.feat_cait_agg
                        cait_img = cait_agg(input_imgs)
                        for ilayer_name in self.trainer_para["cait"]["layers"]:
                            img_layer = cait_img[ilayer_name]

                            if self.trainer_para["cait"]["backbone_name"] in self.trainer_para["cait"]["tr_net"]:
                                patch_shape = int(np.sqrt(img_layer.shape[1]))
                                img_layer = img_layer.view(-1, patch_shape, patch_shape, 192)
                                img_layer = img_layer.permute(0, 3, 1, 2)

                            if img_layer.shape[2] != self.agg_aug_bi_bigger_size:
                                img_layer = F.interpolate(
                                    img_layer,
                                    size=(self.agg_aug_bi_bigger_size, self.agg_aug_bi_bigger_size),
                                    mode="bilinear",
                                    align_corners=False,
                                ) 
                            img_agg_layer_list.append(img_layer)

                    img_agg_input = torch.cat(img_agg_layer_list, 1)

                    if "layer_agg" in self.trainer_para and self.trainer_para["layer_agg"]["mode"] == True:                         
                        batch_size = img_agg_input.shape[0]
                        dim_len = img_agg_input.shape[1]
                        patch_shape = img_agg_input.shape[2]

                        img_agg_input = img_agg_input.permute(0, 2, 3, 1).reshape(-1, dim_len)
                        img_agg_input = self.layer_agg(img_agg_input).reshape(batch_size, patch_shape, patch_shape, -1)
                        img_agg_input = img_agg_input.permute(0, 3, 1, 2)    

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
                        """
                        fea_out, patch_shape = self.patch_make_score._embed(outputs, detech_fea=False, provide_patch_shapes=True)
                        fea_in = self.patch_make_score._embed(img_agg_input, detech_fea=False)
                        
                        bs = fea_out.shape[0]
                        #print(fea_out.size())
                        fea_out = fea_out.view(bs, patch_shape[0], patch_shape[1], -1)
                        fea_in = fea_in.view(bs, patch_shape[0], patch_shape[1], -1)

                        fea_out = fea_out.permute(0, 3, 1, 2)
                        fea_in = fea_in.permute(0, 3, 1, 2)

                        array_inputs  = fea_in.cpu().numpy()
                        array_outputs = fea_out.cpu().numpy()
                        remaps_ori = np.sqrt((array_inputs - array_outputs)**2)
                        remaps = np.mean(remaps_ori, axis=1)
                        max_mean_score = np.mean(np.max(remaps_ori, axis=(2, 3)), axis=1)
                        mean_max_score = np.max(np.mean(remaps_ori, axis=(1)), axis = (-1, -2))
                        lr = 0.6
                        scores = lr * max_mean_score + mean_max_score * (1 - lr)
                         """
                        

                        #abs_fea_score, patch_shape= self.patch_make_score._embed(torch.sqrt((img_agg_input - outputs)**2), provide_patch_shapes=True)
                        #abs_fea_score = np.sqrt(abs_fea_score)

                        fea_out, patch_shape = self.patch_make_score._embed(outputs, provide_patch_shapes=True)
                        fea_in = self.patch_make_score._embed(img_agg_input)
                        abs_fea_score = np.sqrt((fea_out - fea_in)**2)
                        
                        #instance score

                        #alpha_batch_score = 0.1
                        #batch_size = abs_fea_score.shape[0]
                        #local_batch_score = np.expand_dims(np.mean(fea_out, axis = 0), axis=0).repeat(batch_size, axis=0)
                        #local_batch_score = np.sqrt((fea_out - local_batch_score)**2)
                        #abs_fea_score = abs_fea_score - 0.3 * local_batch_score
                      
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

                    elif self.trainer_para["score_method"] == "patch_dis_mem":
                        fea_out, patch_shape = self.patch_make_score._embed(outputs, provide_patch_shapes=True)
                        fea_in = self.patch_make_score._embed(img_agg_input)
                        
                        abs_fea_score = np.sqrt((fea_out - fea_in)**2)
                        
                        #instance score
                        max_mean_score = np.mean(np.max(abs_fea_score, axis=(1)), axis = -1)
                        mean_max_score = np.max(np.mean(abs_fea_score, axis=(-1)), axis = -1)
                        
                        lr = 0.6
                        scores = lr * max_mean_score + mean_max_score * (1 - lr)

                        #pix score
                        fea_score = np.mean(abs_fea_score, axis=-1)
                        remaps = fea_score.reshape(fea_score.shape[0], patch_shape[0], patch_shape[1])    

                        batch_size = fea_in.shape[0]
                        scores_memory = 0
                        for i_fea_mem in fea_memory:
                            i_fea_mem = i_fea_mem.unsqueeze(0)
                            i_bs_fea_mem = i_fea_mem.repeat(batch_size, 1, 1).cpu().numpy()

                            abs_fea_score = np.sqrt((i_bs_fea_mem - fea_in)**2)

                            #instance score
                            max_mean_score = np.mean(np.max(abs_fea_score, axis=(1)), axis = -1)
                            mean_max_score = np.max(np.mean(abs_fea_score, axis=(-1)), axis = -1)

                            lr = 0.6
                            scores_memory += lr * max_mean_score + mean_max_score * (1 - lr)

                            #pix score
                            #fea_score = np.mean(abs_fea_score, axis=-1)
                            #remaps = fea_score.reshape(fea_score.shape[0], patch_shape[0], patch_shape[1]) 
                        scores_memory = scores_memory / fea_memory.shape[0]
                        scores = scores_memory
                            
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
            'image_auroc':image_test_results['auroc']*100.,
            'pixel_auroc':pix_test_results['auroc']*100.,
            'pix_pro':anomaly_pixel_scores["auroc"]*100.
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

        image_dict = {
            "image":ori_imgs_list,
            "mask":mask_gt_list,
            "dist_map":dist_maps_list,
        }

        if self.trainer_para["save_test_image"] == True:
            plot_images(self.out_path + "/test_images/", image_dict, denorm=True)

    
    
    