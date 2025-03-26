import torch

# 学習関数
# model: 学習するモデル
def train(model, optimizer, data_loader, device):  
  loss_ave = 0  
  model.train()  # モデルを学習モードに変更
  
  for imgs, targets in data_loader:  
    # Faster R-CNNは入力がList形式なので、img・targetそれぞれ変換
    img = list(image.to(device, dtype=torch.float) for image in imgs)
    target = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # imgとtargetをモデルに入力し、lossを計算
    loss_dict = model(img, target)
    
    # 4種類のlossが出力されるので合計を算出
    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    loss_ave += loss_value
    
    optimizer.zero_grad()     #勾配リセット  
    losses.backward()          #バックプロパゲーション算出
    optimizer.step()             #重み更新
  
  # Lossの平均を返す
  return loss_ave / len(data_loader)