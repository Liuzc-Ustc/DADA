import torch
from torchvision.transforms import functional as F
import numpy as np
import random
import torch.nn.functional as nnF
import cv2
import math

#% I.待调用的函数： 
## 1.有角度地旋转
def RotateBound(data, angle, inter):
    scale = math.sin(abs(angle) * math.pi / 180) + math.cos(abs(angle) * math.pi / 180)
    # print('scale =',scale)
    (h, w) = data.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    # print("M shape =",M.shape)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # print('nW,nH =',nW,nH)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(data, M, (nW, nH), flags=inter)

## 2.随机裁剪到（128，128，128）
def randomCrop(seis, fault, size=(128, 128, 128)):
    shape = seis[0].shape
    size = np.array(size)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    c = random.randint(0, lim[2])
    return seis[:, w:w + size[0], h:h + size[1], c:c + size[2]], \
        fault[:, w:w + size[0], h:h + size[1], c:c + size[2]]

## 3.高斯模糊（模糊因子的定义 大概在多大的范围，输入地震数据的维度是什么样的？）
def RandomGaussianBlur_h(seismic, sigma_range, p=0.5):
    if random.random() < p:  return seismic
    sig_min, sig_max = sigma_range
    sigma_h = random.uniform(sig_min, sig_max)
    kernel_h = int(np.ceil(sigma_h) * 2 + 1)
    sigma_w = random.uniform(sig_min, sig_max)
    kernel_w = int(np.ceil(sigma_w) * 2 + 1)
    seismic = seismic.permute((0, 2, 1, 3))
    seismic = F.gaussian_blur(seismic, kernel_size=[kernel_h, kernel_w], sigma=[sigma_h, sigma_w])
    seismic = seismic.permute((0, 2, 1, 3))
    return seismic

## 4.高斯模糊（模糊因子的定义 大概在多大的范围，输入地震数据的维度是什么样的？）
def RandomGaussianBlur_w(seismic, sigma_range, p=0.5):
    if random.random() < p:  return seismic
    sig_min, sig_max = sigma_range
    sigma_h = random.uniform(sig_min, sig_max)
    kernel_h = int(np.ceil(sigma_h) * 2 + 1)
    sigma_w = random.uniform(sig_min, sig_max)
    kernel_w = int(np.ceil(sigma_w) * 2 + 1)
    seismic = seismic.permute((0, 3, 2, 1))
    seismic = F.gaussian_blur(seismic, kernel_size=[kernel_h, kernel_w], sigma=[sigma_h, sigma_w])
    seismic = seismic.permute((0, 3, 2, 1))
    return seismic

## 5.高斯模糊（模糊因子的定义 大概在多大的范围，输入地震数据的维度是什么样的？有点像减弱振幅？类似于伽马变换的变暗？对模型训练的影响很大吗？）
def RandomGaussianBlur_t(seismic, sigma_range, p=0.5):
    if random.random() < p:  return seismic
    sig_min, sig_max = sigma_range
    sigma_h = random.uniform(sig_min, sig_max)
    kernel_h = int(np.ceil(sigma_h) * 2 + 1)
    sigma_w = random.uniform(sig_min, sig_max)
    kernel_w = int(np.ceil(sigma_w) * 2 + 1)
    seismic = F.gaussian_blur(seismic, kernel_size=[kernel_h, kernel_w], sigma=[sigma_h, sigma_w])
    return seismic

#% II.数据归一化
#1.数据归一化 至 0-1 （numpy 版本）
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range+1e-6)

##2.Z-Score（即 均值方差） 标准化 + clip裁剪到规定范围  + 归一化至0-1 （numpy 版本）
def z_score_clip(data, clp_s=3.2):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))

##3.Mean-std（即 均值方差） 标准化 
def z_score(data):
    z = (data - np.mean(data)) / np.std(data)
    return z
#% III.数据增广方法：seis shape(ni,nx,nt)
# 1. RandomRotate 概率性 随机旋转
def RandomRotate(seis,fault,p=0.25,ii=None):
    if ii == None:
        i = random.randint(1,3)
    else:
        i = ii
    if random.random() > p:
        seis = np.rot90(seis,i,(1,0))
        fault = np.rot90(fault,i,(1,0))   
        return seis,fault
    return seis,fault 

# 2. RandomInlineFlip 概率性 随机翻转
def RandomInlineFlip(seis,fault,p=0.35):
    i = 1
    if random.random() > p:
        seis = np.flip(seis,i)
        fault = np.flip(fault,i)   
        return seis,fault
    return seis,fault 

# 3. RandomXlineFlip 概率性 随机翻转
def RandomXlineFlip(seis,fault,p=0.35):
    i = 2
    if random.random() > p:
        seis = np.flip(seis,i)
        fault = np.flip(fault,i)   
        return seis,fault
    return seis,fault 

# 4. RandomTlineFlip 概率性 随机翻转
def RandomTlineFlip(seis,fault,p=0.35):
    i = 0
    if random.random() > p:
        seis = np.flip(seis,i)
        fault = np.flip(fault,i)   
        return seis,fault
    return seis,fault 

# 5. RandomRotateAgSynIXline 概率性 随机旋转（沿着 inline或者xline剖面）
def RandomRotateAgSynIXline(seis, fault, p=0.35,ae=None):

    if random.random() < p:
        return seis, fault
    _, cube_size, _ = seis.shape
    if ae == None:  
      angle = random.randint(-30, 30)
    else:
        angle = ae
    seis = RotateBound(seis.transpose((1, 2, 0)), angle, inter=1).transpose((2, 0, 1))
    fault = RotateBound(fault.transpose((1, 2, 0)), angle, inter=cv2.INTER_NEAREST).transpose((2, 0, 1))
    l = int((seis.shape[1] - cube_size) / 2)
    seis = seis[:, l:l + cube_size, l:l + cube_size]
    fault = fault[:, l:l + cube_size, l:l + cube_size]


    return seis,fault

# 6. RandomRotateAgSynTline 概率性 随机旋转（沿着 时间切片）
def RandomRotateAgSynTline(seis, fault, p=0.35,ae=None):
    if random.random() < p:
        return seis, fault
    cube_size, _, _ = seis.shape
    if ae == None:  
      angle = random.randint(-45, 45)
    else:
        angle = ae
    seis = RotateBound(seis, angle, inter=1)
    fault = RotateBound(fault, angle, inter=0)
    l = int((seis.shape[1] - cube_size) / 2)
    seis = seis[l:l + cube_size, l:l + cube_size, :]
    fault = fault[l:l + cube_size, l:l + cube_size, :]
    print(np.min(fault),np.max(fault))
    return seis,fault

#7.RandomSynClip 概率性 随机放大后 再 crop 成128*128*128 
def RandomSynClip(seis, fault, syn_size=128, scale=2.0, p=0.3):
    if random.random() < p:
        return seis, fault
    # fault = np.clip(fault,0,1)
    seis_copy = seis.copy()
    fault_copy = fault.copy()
    seis = torch.tensor(seis_copy)[None,:,:,:].float()
    fault = torch.tensor(fault_copy)[None,:,:,:].float()
    t_scale = random.uniform(1, 3.)
    resize_t = round(syn_size * t_scale)

    i_scale = random.uniform(1, scale)
    resize_i = round(syn_size * i_scale)

    x_scale = random.uniform(1, scale)
    resize_x = round(syn_size * x_scale)

    resize_seis = nnF.interpolate(seis[None], (resize_x, resize_i, resize_t), mode='trilinear')[0]
    resize_fault = (nnF.interpolate(fault[None], (resize_x, resize_i, resize_t), mode='nearest') ).float()[0]
    resize_seis, resize_fault = randomCrop(resize_seis, resize_fault)
    resize_seis,resize_fault = np.squeeze(resize_seis.numpy()),np.squeeze(resize_fault.numpy())
    # resize_fault = np.clip(resize_fault,0,1)
    print(np.min(resize_fault),np.max(resize_fault))
       
    return resize_seis, resize_fault

# 8.## RandomGammaTransfer概率性随机伽马变换（概率性选择 + 亮度变换选择【1 暗变亮，0 亮 变 暗】 + 0-1归一化后变换到 原始数据的范围）
def RandomGammaTransfer(seismic, p=0.25,gamma=None):
    if random.random() < p:  return seismic
    s_max, s_min = np.max(seismic), np.min(seismic)
    if gamma==None:
        if random.randint(0, 1):
            gamma = random.uniform(0.6667, 1)
        else:
            gamma = random.uniform(1, 1.5)
    else:
        gamma = gamma

    print(gamma)
    gamma_seismic = (seismic - s_min) ** gamma

    gamma_range = np.max(gamma_seismic) - np.min(gamma_seismic)
    gamma_seismic = ((gamma_seismic - np.min(gamma_seismic)) / gamma_range) * (s_max - s_min) + s_min
    # gamma_seismic = z_score(gamma_seismic)
    return gamma_seismic

## 9.高斯模糊(变暗)
def RandomGaussianBlur(seismic, p=0.15, sigma_range=[2.7,2.7]):  ## [0.01,1]
    if random.random() < p:  return seismic
    seismic = torch.tensor(seismic)[None,:,:,:].float()
    aug_funcs = [RandomGaussianBlur_w, RandomGaussianBlur_h, RandomGaussianBlur_t]
    # if random.randint(0, 1):
    #     sigma_range = [0.01,0.55]
    # else:
    #     sigma_range = [0.1,0.65]    
    # print(sigma_range)
    random.shuffle(aug_funcs)
    for func in aug_funcs:
        seismic = func(seismic, sigma_range)
    seismic = np.squeeze(seismic.numpy())
    return seismic


## 10. 裁剪数据到（64，64，64）
def cropdata(data, fault):
    start_t = random.randint(0,64)
    start_h = random.randint(0, 64)
    start_w = random.randint(0, 64)

    seis = data[start_t: start_t + 64, start_h:start_h + 64, start_w:start_w + 64]
    fault = fault[start_t:start_t + 64, start_h:start_h + 64, start_w:start_w + 64]
    return seis, fault
# 11. RandomFlip 概率性 随机翻转
def RandomFlip(seis,fault,p=0.35,ii=None):
    if ii == None:
        j = random.randint(0,5)
    else:
        j = ii
    a=[0,1,2,(0,2),(1,2),None,(0,1)]
    i = a[j]
    if random.random() > p:
        seis = np.flip(seis,i)
        fault = np.flip(fault,i)   
        return seis,fault
    return seis,fault 

# 12.RandomGaussianNoise
def RandomGaussianNoise(seismic):
    s_max, s_min = np.max(seismic), np.min(seismic)
    scale = random.uniform(0.1, 0.5) * (s_max - s_min) * 0.1
    noise = np.random.normal(loc=0.0, scale=scale, size=seismic.shape)
    seismic = seismic + noise
    seismic = np.clip(seismic, a_min=s_min, a_max=s_max)
    return seismic