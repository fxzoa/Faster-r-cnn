import torch
import time
import model
from train2 import train
from transforms import get_transform
from CustomDataset import Custom_Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
#バッチデータをまとめる関数
def collate_fn(batch):
  return tuple(zip(*batch))

#メイン関数
if __name__ == '__main__':
  data_file = 'dataset/train/JPEGImages/'
  label_file = 'dataset/train/Annotations/'
  
  transform = get_transform(train=True, resize=1.0, hflip=0.5,vflip=0.5,brightness=0.5,noise=0.5)
  dataset = Custom_Dataset(data_file, transforms=transform)
  
  batch_size = 1
  data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    collate_fn=collate_fn,pin_memory=True)
  num_classes = 2
  net = model.get_instance_model(num_classes)

  print(device)
  net.to(device)
  params = [p for p in net.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
  #3エポックごとにlrを0.5倍
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.5)
  #エポック数
  num_epochs = 10
  loss = 0
  loss_max = 0
  loss_min = 0
  epoch_min = 0
  start_time = time.time()
  for epoch in range(num_epochs):
    loss = train(net, optimizer, data_loader, device)
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
    # モデルデータ保存
    torch.save(net, 'test.pth')
    end_time = time.time()
    print("Epoch[{}] 平均Loss [{}], 最小Loss[{}], 最小Loss_epoch[{}], 学習率 [{}, Total_Time[{}]]".format(epoch, loss, loss_min, epoch_min, lr, end_time - start_time))
    # 学習率の更新
    lr_scheduler.step()

  print("処理時間:{0}".format(end_time - start_time))