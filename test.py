import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import PIL.Image as pil
from torchvision import transforms
import numpy as np
import networks
import cv2
import matplotlib.pyplot as plt


def combine_quadtree(output):
    I = {}
    J = {}
    M = {}
    n = 6
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

    # imgI : disparity with quadtree overlay
    # imgJ : disparity
    return imgI, imgJ



def main(image_path, model_path, crit, device):
    print(model_path)
    with torch.no_grad():
        ## Load model
        # encoder
        encoder_path = os.path.join(model_path, "encoder.pth")
        encoder_dict = torch.load(encoder_path, map_location=torch.device(device))
        width, height = encoder_dict['width'], encoder_dict['height']
        encoder = networks.ResnetEncoder(18, False)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        encoder.to(device)
        encoder.eval()

        # decoder
        decoder_path = os.path.join(model_path, "depth.pth")
        decoder_dict = torch.load(decoder_path, map_location=torch.device(device))
        decoder = networks.NQGN_SparseDecoder()
        decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device(device)))
        decoder.to(device)
        decoder.eval()


        ## load image
        input_image = pil.open(image_path).convert('RGB')
        input_image_resized = input_image.resize((width, height), pil.LANCZOS)
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0).to(device)


        ## Inference
        quad, mask = decoder(encoder(input_image_pytorch.repeat(5,1,1,1)), None, crit=crit)


    output = {}
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

    I, J = combine_quadtree(output)


    vmin = np.min(I)
    vmax = np.max(I)
    plt.figure()
    plt.subplot(311)
    plt.imshow(input_image)
    plt.title("input image")
    plt.subplot(312)
    plt.imshow(J, vmin=vmin, vmax=vmax)
    plt.title("Recombined Quadtree.")
    plt.subplot(313)
    plt.imshow(I, vmin=vmin, vmax=vmax)
    plt.title("Recombined Quadtree with overlay.")


    plt.figure()
    plt.subplot(241)
    plt.imshow(input_image_pytorch[0].permute(1,2,0).cpu())
    plt.title("Input Image")
    plt.subplot(242)
    plt.imshow(output[("disp", 5)][0,0].cpu())
    plt.title("Q5")
    plt.subplot(243)
    plt.imshow(output[("disp", 4)][0,0].cpu())
    plt.title("Q4")
    plt.subplot(244)
    plt.imshow(output[("disp", 3)][0,0].cpu())
    plt.title("Q3")
    plt.subplot(245)
    plt.imshow(J)
    plt.title("Recombined Quadtree")
    plt.subplot(246)
    plt.imshow(output[("disp", 2)][0,0].cpu())
    plt.title("Q2")
    plt.subplot(247)
    plt.imshow(output[("disp", 1)][0,0].cpu())
    plt.title("Q1")
    plt.subplot(248)
    plt.imshow(output[("disp", 0)][0,0].cpu())
    plt.title("Q0")

    plt.show()



if __name__ == '__main__':
    ## Parameters
    image_path = "images/image.jpg"
    model_path = "models/NQGN_comp10"
    crit = 0.04
    device = "cuda"
    main(image_path, model_path, crit, device)

    ## Parameters
    image_path = "images/image.jpg"
    model_path = "models/NQGN_comp30"
    crit = 0.07
    device = "cuda"
    main(image_path, model_path, crit, device)
