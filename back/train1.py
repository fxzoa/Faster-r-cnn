import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob

# カスタムデータセットの作成
#################################################################################
class CustomDataset(Dataset):
  def __init__(self, image_paths, annotations):
    self.image_paths = image_paths
    self.annotations = annotations

  def __getitem__(self, idx):
    img = Image.open(self.image_paths[idx]).convert("RGB")
    img = F.to_tensor(img)
    boxes = torch.tensor(self.annotations[idx]["boxes"], dtype=torch.float32)  # [xmin, ymin, xmax, ymax]
    labels = torch.tensor(self.annotations[idx]["labels"], dtype=torch.int64)
    target = {"boxes": boxes, "labels": labels}
    return img, target

  def __len__(self):
    return len(self.image_paths)
  
  
###################################################################################
# データの設定
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
xml_paths_train = glob("######/*.xml")
xml_paths_val = glob("######/*.xml")
annotations = { }

# データセットとデータローダーの設定
dataset = CustomDataset(image_paths, annotations)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# モデルのロード（最新バージョン）
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
num_classes = 3  # 例: 背景 + 2クラス
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# デバイス設定とトレーニング
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# トレーニングループ
for epoch in range(10):
  model.train()
  for images, targets in data_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
  print(f"Epoch {epoch+1}/10, Loss: {losses.item()}")

# モデルの保存
torch.save(model.state_dict(), "faster_rcnn_model_v2.pth")