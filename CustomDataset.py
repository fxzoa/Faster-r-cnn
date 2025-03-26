import torch
import json
from PIL import Image
import base64
import glob
import io
import copy

class_map = {"background": 0}  # 背景を0として初期化
def get_class_map(class_list=[]): 
  for idx, class_name in enumerate(class_list, start=1):  # 1から始まるインデックスを使用
      class_map[class_name] = idx
    
# データセットクラス
class Custom_Dataset(torch.utils.data.Dataset):
  def __init__(self, root, transforms, classs=["cat", "dog", "car"]):
    self.imgs = []
    self.targets = []
    self.class_map=get_class_map(classs)
    self.transforms = transforms
    self.CreateDataset(root)
    
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, idx):
    img = self.imgs[idx]
    target = copy.deepcopy(self.targets[idx])
    if self.transforms is not None:
      img, target = self.transforms(img, target)
    return img, target
  
  # base64形式をPIL型に変換
  def img_data_to_pil(self,img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil
  
  # jsonデータの読み込み
  def CreateDataset(self, json_dir):
    json_paths = glob.glob(json_dir + '/*.json')
    for json_path in json_paths:
      json_file = open(json_path)
      json_data = json.load(json_file)

      img_b64 = json_data['imageData']           # imageDataをkeyにしてデータを取り出す
      img_data = base64.b64decode(img_b64)  
      img_pil = self.img_data_to_pil(img_data)  # base64形式をPIL型に変換する
      self.imgs.append(img_pil)                       # imgsに画像を追加

      num_objs = len(json_data['shapes'])        # 物体の個数
      boxes = []
      labels = []
      
      # バウンティングボックスから左上、左下、右上、右下の座標を取得し、
      # boxesリストへ追加
      for i in range(num_objs):
        shape = json_data['shapes'][i]
        box =shape['points']
        x_list = []
        y_list = []
        for i in range(len(box)):
          x,y=box[i]
          x_list.append(x)
          y_list.append(y)
        xmin = min(x_list)
        xmax = max(x_list)
        ymin = min(y_list)
        ymax = max(y_list)
        boxes.append([xmin, ymin, xmax, ymax])
        
        # クラスラベルの設定
        class_name = shape['label']                            # JSONからクラス名を取得
        class_id = self.class_map.get(class_name, 0)  # マッピング辞書からIDを取得（不明な場合は0）
        labels.append(class_id)
        
      # boxesリストをtorch.Tensorに変換
      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      labels = torch.as_tensor(labels, dtype=torch.int64)
      
      #targetsへboxesリストとlabelsリストを設定
      target = {}
      target["boxes"] = boxes
      target["labels"] = labels
      self.targets.append(target)