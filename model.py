import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# モデルを取得
# num_classesはデータセットのクラス数を指定します
def get_instance_model(num_classes):
    # COCOデータセットで訓練した、訓練済みモデルをロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 既存の分類器を、ユーザーが定義したnum_classesを持つ新しい分類器に置き換えます
    # 分類器にインプットする特徴量の数を取得
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 事前訓練済みのヘッドを新しいものと置き換える
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes) 
    
    return model