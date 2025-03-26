import torch
import random
import numpy as np
from torchvision.transforms import functional as F

# 画像をTensorに変換  
class ToTensor(object):
  def __call__(self, image, target):
    image = F.to_tensor(image)
    return image, target
    
# リサイズ
class Trans_Resize(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    c, w, h = image.shape
    if self.prob != 1.0:
      image = F.resize(img=image, size=(int(w*self.prob), int(h*self.prob)))
      target["boxes"][:,:] = target["boxes"][:,:] * self.prob
    return image, target
    
#左右反転
class RandomHorizontalFlip(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    if random.random() < self.prob:
      w = image.shape[2]
      image = image.flip(2)
      bbox = target["boxes"]
      bbox[:, [0, 2]] = w - bbox[:, [2, 0]]
      target["boxes"] = bbox
    return image, target
  
#上下反転
class RandomVarticalFlip(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    if random.random() < self.prob:
      h = image.shape[1]
      image = image.flip(1)
      bbox = target["boxes"]
      bbox[:, [1, 3]] = h - bbox[:, [3, 1]]
      target["boxes"] = bbox
    return image, target
    
#明るさ変更(ガンマ補正)
class RandomBrightness(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    if random.random() < self.prob:
      gamma = random.randint(5, 15) * 0.1
      image[:,:,:] = pow(image[:,:,:], 1.0/gamma)
    return image, target
    
#ランダムノイズ
class RandomNoise(object):
  def __init__(self, prob):
    self.prob = prob
  def __call__(self, image, target):
    if random.random() < self.prob:
      height, width = image.shape[-2:]
      pts_count = np.random.randint((height*width)/8, (height*width)/4, 1)
      r = torch.as_tensor(np.random.rand(pts_count.item()), dtype=torch.float32)
      g = torch.as_tensor(np.random.rand(pts_count.item()), dtype=torch.float32)
      b = torch.as_tensor(np.random.rand(pts_count.item()), dtype=torch.float32)
      pts_rgb = torch.reshape(torch.cat([r,g,b],dim=0), (3,pts_count.item()))
      pts_x = np.random.randint(0, width-1 , pts_count.item())
      pts_y = np.random.randint(0, height-1, pts_count.item())
      image[:,pts_y,pts_x] = pts_rgb
    return image, target
    
#複数の前処理をまとめる
class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms
    
  def __call__(self, image, target):
    for t in self.transforms:
      image, target = t(image, target)
    return image, target
      

####################################################################
# 画像の前処理
# PIL imageをPyTorch Tensorに変換
def get_transform(train=False, resize=1.0, hflip=0.0, vflip=0.0, brightness=0.0,  noise=0.0):
  transforms = []
  transforms.append(ToTensor())
  transforms.append(Trans_Resize(resize))
  if train:         #学習時のみの処理を追加
    transforms.append(RandomHorizontalFlip(hflip))
    transforms.append(RandomVarticalFlip(vflip))
    transforms.append(RandomBrightness(brightness))
    transforms.append(RandomNoise(noise))
  return Compose(transforms)