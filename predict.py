import torch
import torchvision
from torchvision.models.detection import FasterRCNN

from PIL import ImageDraw
from lib.CustomDataset import Dataset
from lib.CustomTransforms import get_transform


# 許可リストにFasterRCNNを追加
torch.serialization.add_safe_globals([FasterRCNN])

# GPUが使える場合はGPUを使う
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

val_data_file = 'dataset/val/JPEGImages/'
dataset = Dataset(val_data_file, get_transform())
img, _ = dataset[0]

#GPUで学習したモデルをCPUで読み込み場合は「map_loacation」をcpuに変更
net = torch.load('test.pth', map_location=torch.device('cpu'), weights_only=False)
net.to(device)
net.eval()# モデルを評価モードに変更します

with torch.no_grad():
  prediction = net([img.to(device)])

print(len(prediction))
print(prediction[0]["boxes"][0])
img = torchvision.transforms.functional.to_pil_image(img)

# バウンティングボックス描画
for i in range(len(prediction[0]["boxes"])):
  draw = ImageDraw.Draw(img)
  x = prediction[0]['boxes'][i][0]
  y = prediction[0]['boxes'][i][1]
  w = prediction[0]['boxes'][i][2]
  h = prediction[0]['boxes'][i][3]
  draw.rectangle((x,y,w,h), outline=(255, 0, 0), width=3)

img.show() 