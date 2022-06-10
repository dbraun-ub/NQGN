from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
# from networks import ModelBuilder
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.image


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def densify_disp(outputs):
    # densify disparity maps
    up = nn.Upsample(scale_factor=2, mode='nearest')
    outputs[("disp_dense", 5)] = outputs[("disp", 5)].clone()

    num_elem = torch.sum(1 - torch.round(outputs[("quad_mask", 4)]))

    disp4 = outputs[("disp", 4)].clone()
    mask4 = torch.round(up(outputs[("quad_mask", 4)]))
    disp4 = disp4 * mask4 + up(outputs[("disp", 5)]) * (1 - mask4)

    q5 = torch.sum(1 - mask4) / torch.numel(mask4)

    num_elem += torch.sum(mask4 * (1 - torch.round(outputs[("quad_mask", 3)])))

    outputs[("disp_dense", 4)] = disp4

    disp3 = outputs[("disp", 3)].clone()
    mask3 = torch.round(up(outputs[("quad_mask", 3)]))
    disp3 = disp3 * mask3 + up(disp4) * (1 - mask3)

    q4 = torch.sum(up(mask4) - mask3) / torch.numel(mask3)

    num_elem += torch.sum(mask3 * (1 - torch.round(outputs[("quad_mask", 2)])))

    outputs[("disp_dense", 3)] = disp3

    disp2 = outputs[("disp", 2)].clone()
    mask2 = torch.round(up(outputs[("quad_mask", 2)]))
    disp2 = disp2 * mask2 + up(disp3) * (1 - mask2)

    q3 = torch.sum(up(mask3) - mask2) / torch.numel(mask2)

    num_elem += torch.sum(mask2 * (1 - torch.round(outputs[("quad_mask", 1)])))

    outputs[("disp_dense", 2)] = disp2

    disp1 = outputs[("disp", 1)].clone()
    mask1 = torch.round(up(outputs[("quad_mask", 1)]))
    disp1 = disp1 * mask1 + up(disp2) * (1 - mask1)

    q2 = torch.sum(up(mask2) - mask1) / torch.numel(mask1)

    num_elem += torch.sum(mask1 * (1 - torch.round(outputs[("quad_mask", 0)])))

    outputs[("disp_dense", 1)] = disp1

    disp0 = outputs[("disp", 0)].clone()
    mask0 = torch.round(up(outputs[("quad_mask", 0)]))
    disp0 = disp0 * mask0 + up(disp1) * (1 - mask0)

    q1 = torch.sum(up(mask1) - mask0) / torch.numel(mask0)

    q0 = torch.sum(mask0) / torch.numel(mask0)

    num_elem += torch.sum(mask0)

    outputs[("disp_dense", 0)] = disp0

    return outputs, [q0,q1,q2,q3,q4,q5]


def save_images(output, n, path, id):
    I = {}
    J = {}
    M = {}
    for i in range(n-1,-1,-1):
        I[i] = F.interpolate(output[("disp",i)], scale_factor= 2**i).squeeze()[0].cpu().numpy()
        J[i] = np.copy(I[i])

        j = 0
        while j < 640 + 2**i:
            cv2.line(I[i], (0,j), (640,j), 0, 1)
            cv2.line(I[i], (j,0), (j,640), 0, 1)
            j += 2**i

        if i < n-1:
            M[i] = F.interpolate(torch.round(output[("quad_mask",i)]), scale_factor= 2**(i+1)).squeeze()[0].cpu().numpy()

        if i < n-2:
            M[i] *= M[i+1]

    M[n-1] = 1 - M[n-2]


    imgI = np.zeros((6,192,640))
    imgJ = np.copy(imgI)

    for i in range(n):
        imgI[i] = I[i] * M[i]
        imgJ[i] = J[i] * M[i]
        for j in range(n):
            if j < i:
                imgI[i] *= (1 - M[j])
                imgJ[i] *= (1 - M[j])

    imgI = np.sum(imgI,0)
    imgJ = np.sum(imgJ,0)


    matplotlib.image.imsave(path + '_disp_quadtree_' + '{:04}'.format(id) + '.png', imgI, cmap="magma")
    matplotlib.image.imsave(path + '_disp' + '{:04}'.format(id) + '.png', imgJ, cmap="magma")



def compute_mask_quadtree(disp, max_depth_level, opt):
    # scale disp
    disp, _ = disp_to_depth(disp, opt.min_depth, opt.max_depth)
    b,_,h,w = disp.size()
    mask = {}

    for i in range(max_depth_level, 0, -1):
        s = 2**i
        a = torch.zeros(b, s**2, h//s, w//s)


        for j in range(s**2):
            a[:,j,:,:] = disp[:, 0, (j//s)::s, (j%s)::s]

        mask[i] = (torch.max(a, axis=1).values - torch.min(a, axis=1).values) > opt.crit
        mask[i] = torch.unsqueeze(mask[i],1).type(torch.FloatTensor).to(disp.device)

        if i < max_depth_level:
            mask[i] = mask[i] * F.interpolate(mask[i+1], scale_factor=2.0, mode="nearest")

        assert ((mask[i] >= 0.) * (mask[i] <= 1.)).all()

    return mask

def densify_full_disp(disp, mask):
    # init
    outputs = {}
    q5 = F.interpolate(disp, scale_factor=1/2**5, mode="bilinear")
    outputs[("disp",5)] = q5
    q5 = F.interpolate(q5, scale_factor=2**5)
    quad_disp = q5
    outputs[("disp_dense",5)] = quad_disp

    # loop
    for i in range(4,-1,-1):
        q = F.interpolate(disp, scale_factor=1/2**i, mode="bilinear")
        q = F.interpolate(q, scale_factor=2**i)
        m = F.interpolate(mask[i+1], scale_factor=2**(i+1)).type(torch.bool)
        quad_disp[m] = q[m]
        outputs[("disp_dense", i)] = quad_disp
        outputs[("quad_mask",i)] = mask[i+1]
        outputs[("disp",i)] = F.interpolate(q*m, scale_factor=1/2**i)

    return outputs

def compute_depth_gt_quadtree(depth_gt, max_depth_level, crit):
    disp = 1 / depth_gt


    ## Compute masks
    b,_,h,w = disp.size()
    mask = {}
    gt_masks = {}

    for i in range(max_depth_level, 0, -1):
        s = 2**i
        s2 = 2**(i+1)
        a = torch.zeros(b, s2, h//s, w//s)


        for j in range(s2):
            a[:,j,:,:] = disp[:, 0, (j//s)::s, (j%s)::s]

        assert torch.all(a > 0)

        mask[i] = (torch.max(a, axis=1).values - torch.min(a, axis=1).values) > crit
        mask[i] = torch.unsqueeze(mask[i],1).type(torch.FloatTensor).to(disp.device)

        gt_masks[i] = mask[i]

        if i < max_depth_level:
            mask[i] = mask[i] * F.interpolate(mask[i+1], scale_factor=2.0, mode="nearest")


    ## Construct quadtree
    # init
    outputs = {}
    q5 = F.interpolate(disp, scale_factor=1/2**5, mode="bilinear")
    outputs[("disp",5)] = q5
    q5 = F.interpolate(q5, scale_factor=2**5)
    quad_disp = q5
    outputs[("disp_dense",5)] = quad_disp

    # loop
    for i in range(4,-1,-1):
        q = F.interpolate(disp, scale_factor=1/2**i, mode="bilinear")
        q = F.interpolate(q, scale_factor=2**i)
        m = F.interpolate(mask[i+1], scale_factor=2**(i+1)).type(torch.bool)
        quad_disp[m] = q[m]
        outputs[("disp_dense", i)] = quad_disp
        outputs[("quad_mask",i)] = mask[i+1]
        outputs[("disp",i)] = F.interpolate(q*m, scale_factor=1/2**i)

    num_elem = torch.numel(outputs[("disp", 0)]) - 3 * (torch.sum(1 - torch.round(mask[5]))
                                                     + torch.sum(1 - torch.round(mask[4]))
                                                     + torch.sum(1 - torch.round(mask[3]))
                                                     + torch.sum(1 - torch.round(mask[2]))
                                                     + torch.sum(1 - torch.round(mask[1])))

    num_elem /= outputs[("disp", 0)].size(0)

    return 1 / outputs[("disp_dense", 0)], num_elem, gt_masks





def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 3
    MAX_DEPTH = 80


    if not torch.cuda.is_available() or opt.no_cuda:
        device = "cpu"
    else:
        device = "cuda"


    print(device)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    # with torch.cuda.device(0):
    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location=torch.device(device))

        dataset = datasets.KITTI2012Dataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)

        if opt.monodepth_evaluation:
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        else:
            depth_decoder = networks.NQGN_SparseDecoder()




        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device(device)))

        encoder.to(device)
        encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()


        pred_disps = []
        num_elements = []
        num_elements_gt = []
        gt_depths = []
        gt_depths_dense = []
        q5s = []
        q4s = []
        q3s = []
        q2s = []
        q1s = []
        q0s = []
        pred_masks_0 = []
        pred_masks_1 = []
        pred_masks_2 = []
        pred_masks_3 = []
        pred_masks_4 = []
        gt_masks_0 = []
        gt_masks_1 = []
        gt_masks_2 = []
        gt_masks_3 = []
        gt_masks_4 = []

        count = 0

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(device)
                depth_gt = data["depth_gt"].to(device)
                depth_gt = F.interpolate(depth_gt, size=(opt.height, opt.width))
                quadtree_depth_gt, num_elem_gt, gt_mask = compute_depth_gt_quadtree(depth_gt, 5, opt.ref_crit)


                gt_masks_0.append(gt_mask[1][:,0].cpu().numpy())
                gt_masks_1.append(gt_mask[2][:,0].cpu().numpy())
                gt_masks_2.append(gt_mask[3][:,0].cpu().numpy())
                gt_masks_3.append(gt_mask[4][:,0].cpu().numpy())
                gt_masks_4.append(gt_mask[5][:,0].cpu().numpy())

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                if opt.monodepth_evaluation:
                    features = encoder(input_color)
                    outputs = depth_decoder(features)
                else:
                    features = encoder(input_color)
                    quad, mask = depth_decoder(features, None, crit=opt.crit)
                    output = {}

                if opt.monodepth_evaluation:
                    mask = compute_mask_quadtree(outputs[("disp", 0)], 5, opt)
                    output = densify_full_disp(outputs[("disp", 0)], mask)



                    output, Q = densify_disp(output)
                    num_elem = torch.numel(output[("disp", 0)]) - 3 * (torch.sum(1 - torch.round(mask[5]))
                                                                     + torch.sum(1 - torch.round(mask[4]))
                                                                     + torch.sum(1 - torch.round(mask[3]))
                                                                     + torch.sum(1 - torch.round(mask[2]))
                                                                     + torch.sum(1 - torch.round(mask[1])))
                    num_elem /= output[("disp", 0)].size(0)

                    if count % 10 == 0 and opt.save_images is True:
                        save_images(output, 6, '/media/HDD1/train_daniel/images/' + opt.model_name, count)
                        color = input_color.permute(0,2,3,1).cpu().detach().numpy()[0] * 255
                        color = color.astype(np.uint8)
                        matplotlib.image.imsave("/media/HDD1/train_daniel/images/image_" + '{:04d}'.format(count) + '.png', color)

                else:
                    output[("disp", 0)] = quad[5]
                    output[("disp", 1)] = quad[4]
                    output[("disp", 2)] = quad[3]
                    output[("disp", 3)] = quad[2]
                    output[("disp", 4)] = quad[1]
                    output[("disp", 5)] = quad[0]
                    output[("quad_mask", 0)] = mask[4]
                    output[("quad_mask", 1)] = mask[3]
                    output[("quad_mask", 2)] = mask[2]
                    output[("quad_mask", 3)] = mask[1]
                    output[("quad_mask", 4)] = mask[0]

                    output, Q = densify_disp(output)
                    num_elem = torch.numel(quad[5]) - 3 * (torch.sum(1 - torch.round(mask[4]))
                                                        + torch.sum(1 - torch.round(mask[3]))
                                                        + torch.sum(1 - torch.round(mask[2]))
                                                        + torch.sum(1 - torch.round(mask[1]))
                                                        + torch.sum(1 - torch.round(mask[0])))
                    num_elem /= quad[5].size(0)




                    if count % 2 == 0 and opt.save_images is True:
                        save_images(output, 6, '/media/HDD1/train_daniel/images/' + opt.model_name, count)
                        color = input_color.permute(0,2,3,1).cpu().detach().numpy()[0] * 255
                        color = color.astype(np.uint8)
                        matplotlib.image.imsave("/media/HDD1/train_daniel/images/image_" + '{:04d}'.format(count) + '.png', color)
                        # print(f"depth_gt: {depth_gt.cpu()[0, 0, :].size()}")
                        matplotlib.image.imsave("/media/HDD1/train_daniel/images/ref_" + '{:04d}'.format(count) + '.png', 1/depth_gt.cpu()[0, 0, :].numpy(), cmap="magma")

                count += 1
                pred_disp, _ = disp_to_depth(output[("disp_dense", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                quadtree_depth_gt = quadtree_depth_gt.cpu()[:, 0].numpy()
                depth_gt_dense = depth_gt.cpu()[:, 0].numpy()


                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                num_elements.append(num_elem.cpu().numpy())
                num_elements_gt.append(num_elem_gt.cpu().numpy())
                gt_depths.append(quadtree_depth_gt)
                gt_depths_dense.append(depth_gt_dense)
                q5s.append(Q[5].cpu().numpy())
                q4s.append(Q[4].cpu().numpy())
                q3s.append(Q[3].cpu().numpy())
                q2s.append(Q[2].cpu().numpy())
                q1s.append(Q[1].cpu().numpy())
                q0s.append(Q[0].cpu().numpy())

                pred_masks_0.append(output[("quad_mask", 0)][:,0].cpu().numpy())
                pred_masks_1.append(output[("quad_mask", 1)][:,0].cpu().numpy())
                pred_masks_2.append(output[("quad_mask", 2)][:,0].cpu().numpy())
                pred_masks_3.append(output[("quad_mask", 3)][:,0].cpu().numpy())
                pred_masks_4.append(output[("quad_mask", 4)][:,0].cpu().numpy())


        pred_disps = np.concatenate(pred_disps)
        num_elements = np.mean(num_elements)
        num_elements_gt = np.mean(num_elements_gt)
        gt_depths = np.concatenate(gt_depths)
        gt_depths_dense = np.concatenate(gt_depths_dense)
        q5s = np.mean(q5s)
        q4s = np.mean(q4s)
        q3s = np.mean(q3s)
        q2s = np.mean(q2s)
        q1s = np.mean(q1s)
        q0s = np.mean(q0s)

        pred_masks_0 = np.concatenate(pred_masks_0)
        pred_masks_1 = np.concatenate(pred_masks_1)
        pred_masks_2 = np.concatenate(pred_masks_2)
        pred_masks_3 = np.concatenate(pred_masks_3)
        pred_masks_4 = np.concatenate(pred_masks_4)

        gt_masks_0 = np.concatenate(gt_masks_0)
        gt_masks_1 = np.concatenate(gt_masks_1)
        gt_masks_2 = np.concatenate(gt_masks_2)
        gt_masks_3 = np.concatenate(gt_masks_3)
        gt_masks_4 = np.concatenate(gt_masks_4)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()


    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    errors_dense = []
    ratios = []
    values_pred = []
    values_gt = []
    pred_depth_full_array = []
    MLE_mask = []
    MLE_mask_2 = []


    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_depth_dense = gt_depths_dense[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_depth_full = 1 / pred_disp
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))#, interpolation=cv2.INTER_NEAREST)
        pred_depth = 1 / pred_disp



        mask = (gt_depth > 0)

        # prediction evaluation
        pred_depth = pred_depth[mask] * STEREO_SCALE_FACTOR
        gt_depth = gt_depth[mask] * STEREO_SCALE_FACTOR
        gt_depth_dense = gt_depth_dense[mask] * STEREO_SCALE_FACTOR

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        pred_depth_full[pred_depth_full < MIN_DEPTH] = MIN_DEPTH
        pred_depth_full[pred_depth_full > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))
        errors_dense.append(compute_errors(gt_depth_dense, pred_depth))


        values_pred.append(pred_depth)
        values_gt.append(gt_depth)


        MLE_mask.append((np.sum(gt_masks_0[i] * pred_masks_0[i] + (1 - gt_masks_0[i]) * (1 - pred_masks_0[i]))
                        + np.sum(gt_masks_1[i] * pred_masks_1[i] + (1 - gt_masks_1[i]) * (1 - pred_masks_1[i]))
                        + np.sum(gt_masks_2[i] * pred_masks_2[i] + (1 - gt_masks_2[i]) * (1 - pred_masks_2[i]))
                        + np.sum(gt_masks_3[i] * pred_masks_3[i] + (1 - gt_masks_3[i]) * (1 - pred_masks_3[i]))
                        + np.sum(gt_masks_4[i] * pred_masks_4[i] + (1 - gt_masks_4[i]) * (1 - pred_masks_4[i])))
                        / (gt_masks_0[i].size + gt_masks_1[i].size + gt_masks_2[i].size + gt_masks_3[i].size + gt_masks_4[i].size))

        MLE_mask_2.append((np.sum(gt_masks_0[i] * pred_masks_0[i])
                        + np.sum(gt_masks_1[i] * pred_masks_1[i])
                        + np.sum(gt_masks_2[i] * pred_masks_2[i])
                        + np.sum(gt_masks_3[i] * pred_masks_3[i])
                        + np.sum(gt_masks_4[i] * pred_masks_4[i])
                        / (gt_masks_0[i][gt_masks_0[i] == 1].size
                            + gt_masks_1[i][gt_masks_1[i] == 1].size
                            + gt_masks_2[i][gt_masks_2[i] == 1].size
                            + gt_masks_3[i][gt_masks_3[i] == 1].size
                            + gt_masks_4[i][gt_masks_4[i] == 1].size)))


        pred_depth_full_array.append(np.vstack(pred_depth_full))




    mean_errors = np.array(errors).mean(0)
    mean_errors_dense = np.array(errors_dense).mean(0)

    MLE_global = np.mean(MLE_mask)
    MLE_global_2 = np.mean(MLE_mask_2)



    print("\n  " + ("{:>8} | " * 11).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "num_elem", "compression rate", "GT num_elem", "GT compression rate"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist())
        + ("&{: 8.1f}  ").format(num_elements) + (" &{: 8.1f} ").format(640*192 / num_elements)
        + ("&{: 8.1f}  ").format(num_elements_gt) + (" &{: 8.1f} ").format(640*192 / num_elements_gt) + "\\\\")
    print(("&{: 8.3f}  " * 7).format(*mean_errors_dense.tolist())
        + ("&{: 8.1f}  ").format(num_elements) + (" &{: 8.1f} ").format(640*192 / num_elements)
        + ("&{: 8.1f}  ").format(192*640) + (" &{: 8.1f} ").format(1) + "\\\\")

    print("\n  " + ("{:>8} | " * 6).format("Q5", "Q4", "Q3", "Q2", "Q1", "Q0"))
    print(("&{: 8.4f}  " * 6).format(q5s, q4s, q3s, q2s, q1s, q0s))

    print(f"MLE: {MLE_global}")
    print(f"MLE 2: {MLE_global_2}")

    with open(f"results/table_results.txt", 'a') as f:
        f.write(("{:>8} | " * 8).format("Method", "crit", "comp. rate", "abs_rel quad", "sq_rel quad", "rmse quad", "abs_rel dense", "sq_rel dense", "rmse dense"))
        f.write("\n" + opt.model_name + " & " + str(opt.crit) + (" &{: 8.1f} ").format(640*192 / num_elements)
                + (" &{: 8.3f} " * 3).format(*mean_errors[:3].tolist()) +  (" &{: 8.3f} " * 3).format(*mean_errors_dense[:3].tolist())
                + (" &{: 8.1f} ").format(640*192 / num_elements_gt) + "\\\\")

        f.write("\n  " + ("{:>8} | " * 6).format("Q5", "Q4", "Q3", "Q2", "Q1", "Q0"))
        f.write(("&{: 8.4f}  " * 6).format(q5s, q4s, q3s, q2s, q1s, q0s))



if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
