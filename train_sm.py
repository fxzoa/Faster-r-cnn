import torch
import time, os, argparse
from lib.CustomModel import get_instance_model
from lib.CustomTrain import train
from lib.CustomTransforms import get_transform
from lib.CustomDataset import Dataset

# デバイスの設定
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# データセットのパスを取得
sm_dataset_dir = os.path.join(os.environ['SM_INPUT_DIR'], 'data', 'training')
datasets = os.path.join(sm_dataset_dir, 'dataset')
print(f'sagemaker_dataset_base_dir: {datasets}')

# トレーニング済みモデル保存先
model_dir = os.path.join(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
model_file = os.path.join(model_dir, "faster_r_cnn_best.pt")

#バッチデータをまとめる関数
def collate_fn(batch):
  return tuple(zip(*batch))

# SageMakerのデフォルトの環境変数を使用
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--learning-rate', type=float, default=0.005)
  parser.add_argument('--img_size', type=int, default=640)    
  return parser.parse_args()


##############################################################################
#メイン関数
if __name__ == '__main__':
  args = parse_args()
  classs=["cat", "dog", "car"]
  
  # データセットを取得
  transform = get_transform(train=True, resize=1.0, hflip=0.5,vflip=0.5,brightness=0.5,noise=0.5)
  dataset = Dataset(datasets, transforms=transform, classs=classs)
  
  # データローダーを取得
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn,pin_memory=True)
  
  # モデルを取得
  FasterRCNN = get_instance_model(len(classs)+1)
  FasterRCNN.to(device)

  # パラメータを取得
  params = [p for p in FasterRCNN.parameters() if p.requires_grad]

  # 最適化アルゴリズムを設定
  optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
  
  #エポックごとに学習率(lr)を0.5倍
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.5)

  loss = 0
  loss_max = 0
  loss_min = 0
  epoch_min = 0
  start_time = time.time()
  for epoch in range(args.epochs):
    loss = train(FasterRCNN, optimizer, data_loader, device)
    lr=optimizer.param_groups[0]["lr"]
    if epoch == 0:
      loss_max = loss
      loss_min = loss
    else :
      if loss_max < loss:
        loss_max = loss
      elif loss_min > loss:
        loss_min = loss
        epoch_min = epoch

    # 学習率の更新
    end_time = time.time()
    print("Epoch[{}] 平均Loss [{}], 最小Loss[{}], 最小Loss_epoch[{}], 学習率 [{}, Total_Time[{}]]".format(epoch, loss, loss_min, epoch_min, lr, end_time - start_time))
    lr_scheduler.step()

  print("総トレーニング時間:{0}".format(end_time - start_time))
  
  # モデルデータ保存
  torch.save(FasterRCNN, model_file)
 