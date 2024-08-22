# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io

import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TestMeter

from fvcore.nn import FlopCountAnalysis, parameter_count_table



logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, times, pos_tmp, video_idx, meta) in enumerate(test_loader):
        ###############################
        B,C,T,H,W = inputs.shape
        
        pos_embeds = pos_tmp[0].unsqueeze(0)
        for i in range(1, B):
            pos_embeds = torch.cat((pos_embeds, pos_tmp[1].unsqueeze(0)), dim=0)
        # pos_embeds = torch.cat((pos_tmp[0].unsqueeze(0), pos_tmp[1].unsqueeze(0)), dim=0).unsqueeze(1)
        
        avg = torch.nn.AvgPool2d(16,16)
        
        
        import matplotlib.pyplot as plt
        pos_avg = avg(pos_embeds.float())
        
        values, indices = pos_avg.flatten(2).topk(1, dim=2)
        
        x_index = indices // (H//16)
        y_index = indices % (H//16)
        
        num_weight = cfg.BG_WEIGHT / 10.0
        inputs_masks = torch.ones((B,1,T,H,W)) * num_weight
        # inputs_masks = torch.ones((B,1,T,H,W))
        for bi in range(B):
            for ti in range(T):
                x = x_index[bi][ti] + 1
                y = y_index[bi][ti] + 1
                
                # x [1 - 14]
                # y [1 - 14]
                xs = (x-cfg.BG_SIZE) if (x - cfg.BG_SIZE) > 0 else 1
                xe = (x+cfg.BG_SIZE) if (x + cfg.BG_SIZE) < (H//16) else (H//16)
                ys = (y-cfg.BG_SIZE) if (y - cfg.BG_SIZE) > 0 else 1
                ye = (y+cfg.BG_SIZE) if (y + cfg.BG_SIZE) < (H//16) else (H//16)
                
                xs = xs*16 - 1
                xe = xe*16 - 1
                ys = ys*16 - 1
                ye = ye*16 - 1
                
                inputs_masks[bi][0][ti][xs:xe, ys:ye] = 1
                ################################################
        
        # exit()  
        inputs_masks = inputs_masks.repeat((1,C,1,1,1))     
        inputs = inputs * inputs_masks
        ###################################################
        # # print(cur_iter)
        # imgs = inputs[0].permute(1,2,3,0).clone()
        # index = 0
        # for img in imgs:
        #     im = img.detach().cpu().numpy()
        #     name = f"vis/mask/{cur_iter}_{index}_img.png"
        #     cv2.imwrite(name,im)
        #     index += 1
        
        # frames = inputs_masks[0].permute(1,2,3,0).clone()
        # # print(frames[0])
        # index = 0
        # for i,_ in enumerate(imgs):
        #     im = imgs[i] * frames[i]
            
        #     # print(imgs[i])
        #     # print(frames[i])
        #     # print(imgs[i] * frames[i])
        #     # exit()
            
        #     im = im.detach().cpu().numpy()
        #     name = f"vis/mask/{cur_iter}_{index}_mask_origin.png"
        #     cv2.imwrite(name,im)
        #     index += 1
        # print(im)
        
        # # print(imgs[0])
        # # print(frames[0])
        # exit()
        
        # index = 0
        # for frame in frames:
        #     img = frame.detach().cpu().numpy() * 255
        #     name = f"vis/mask/{cur_iter}_{index}_mask.png"
        #     cv2.imwrite(name,img)
        #     index += 1
        #################################
       
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.

            # flop = FlopCountAnalysis(model, (inputs, times))
            # print(flop.total())
            # print(parameter_count_table(model, 5))
            # exit()

            preds = model(inputs, times)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            with open("labels_preds.txt", "a+") as fp:
                print(video_idx, end=" ", file=fp)
                print(labels, end=" ", file=fp)
                print(preds, file=fp)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)
            
        ####
        # print(all_preds.shape)
        # print(all_labels.shape)
        # exit()
        # torch.save(all_preds, "all_preds.pt")
        # torch.save(all_labels, "all_labels.pt")
        # exit()
        ####

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_labels, all_preds], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)
    
    ############### 可视化位置编码 ####################
    # import matplotlib.pyplot as plt
    # pos_embed_img = np.zeros((196, 196))
    # # print(pos_embed_img.shape)
    # for key in model.state_dict():
    #     if "pos_embed" in key:
    #         pos_embed = model.state_dict()[key][0][1:, ...]
    #         for i in range(pos_embed.shape[0]):
    #             pos_embed_im0 = np.zeros((196))
    #             for j in range(pos_embed.shape[0]):
    #                 pos_embed_img[i][j] = torch.cosine_similarity(pos_embed[i][None,], pos_embed[j][None,])
    #                 pos_embed_im0[j] = torch.cosine_similarity(pos_embed[i][None, ], pos_embed[j][None, ])
                
    #             pos_embed_im0 = pos_embed_im0.reshape((14, 14))
    #             plt.imshow(pos_embed_im0)
    #             plt.colorbar()
    #             plt.savefig(f"vis/pos_embed_{i}.png")
    #             plt.close()
                
    
    # plt.imshow(pos_embed_img)
    # plt.colorbar()
    # plt.savefig("pos_embed.png")
    # plt.close()
            
            
    ##################################################
    ################ 可视化时间编码 ###################
    
    # module.model.time_embed
    
    # time_embed_img = torch.zeros((8, 8))
    # for key in model.state_dict():
    #     if "module.model.time_embed" in key:
    #         time_embed = model.state_dict()[key][0]
    #         for i in range(time_embed.shape[0]):
    #             for j in range(time_embed.shape[0]):
    #                 time_embed_img[i][j] = torch.cosine_similarity(time_embed[i][None, ], time_embed[j][None, ])
                    
    # plt.imshow(time_embed_img)
    # plt.colorbar()
    # plt.savefig("time_embed.png")
    # plt.close()

    #################################################
    
    # ################ 可视化帧时序编码 1 ##################
    # frame_time_embed_img = torch.zeros((15, 15))
    # index = 0
    # for key in model.state_dict():
    #     if "frame_time_embedding" in key:
    #         frame_time_embed = model.state_dict()[key]
    #         for i in range(frame_time_embed.shape[0]):
    #             for j in range(i-7, i+8):
    #                 if j >= 0 and j < frame_time_embed.shape[0]:
    #                     # print(f"{i}_{j}")
    #                     # print(frame_time_embed[head].shape)
    #                     # print(frame_time_embed[head][i].shape)
    #                     # print(frame_time_embed[head])
    #                     # print(frame_time_embed[head][i])
    #                     # exit()
    #                     frame_time_embed_img[i][j] = torch.cosine_similarity(frame_time_embed[i][None, ], frame_time_embed[j][None, ])
    #                     # frame_time_embed_img[i][j] = torch.abs(frame_time_embed[i].mean() - frame_time_embed[j].mean())
    #                     # print(frame_time_embed[i].shape)
                        
       
    #         plt.imshow(frame_time_embed_img)
    #         plt.colorbar()
    #         plt.savefig(f"vis/frame_time_embed_{index}.png")
    #         plt.close()
    #         index += 1
    # exit()
    # #################################################
    
    ################ 可视化帧时序编码 2 ##################
    # frame_time_embed_img = torch.zeros((15, 15))
    # index = 0
    # for key in model.state_dict():
    #     if "frame_time_embedding" in key:
    #         frame_time_embed = model.state_dict()[key].permute(1,0)
    #         for head in range(frame_time_embed.shape[0]):
    #             for i in range(frame_time_embed.shape[1]):
    #                 for j in range(i-7, i+8):
    #                     if j >= 0 and j < frame_time_embed.shape[1]:
    #                         frame_time_embed_img[i][j] = torch.abs(frame_time_embed[head][i:i+1][None] - frame_time_embed[head][j:j+1][None])
       
    #             plt.imshow(frame_time_embed_img)
    #             plt.colorbar()
    #             plt.savefig(f"vis/frame_time_embed_{index}_{head}.png")
    #             plt.close()
    #         index += 1
    # exit()
    #################################################
    
    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
