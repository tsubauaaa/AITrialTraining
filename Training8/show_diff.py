import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.utils import save_image

print("loading models")
# 学習済みモデルの読み込み
model_old = models.vgg16(pretrained=False)
model_old.classifier[6] = nn.Linear(in_features=4096, out_features=10)
model_old.load_state_dict(torch.load("model_old_ft.pt"))

# 学習済みモデルの読み込み
model_new = models.resnet18(pretrained=False)
num__new_model_features = model_new.fc.in_features
model_new.fc = nn.Linear(num__new_model_features, 10)
model_new.load_state_dict(torch.load("model_new_ft.pt"))

# datasetの読み込み
print("loading dataset")

with open("data_train.pkl", "rb") as f:
    data_train = pickle.load(f)

with open("data_val.pkl", "rb") as f:
    data_val = pickle.load(f)

batch_size = 1
train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=batch_size,
    shuffle=False,
)
dataloaders = {"train": train_loader, "val": val_loader}

use_gpu = torch.cuda.is_available()
if use_gpu:
    model_old = model_old.cuda()
    model_new = model_new.cuda()

# confusion matrix作成用リスト
old_matrix = []
new_matrix = []
label_matrix = []
print("Inferring")
# 推論処理 (ミニバッチ)
i = 0
model_new.eval()
model_old.eval()
for data in dataloaders["val"]:
    # if i == 1000:
    #     break
    inputs, labels = data  # ImageFolderで作成したデータは、データをラベルを持ってくれます。
    # GPUを使わない場合不要
    if use_gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # ~~~~~~~~~~~~~~forward~~~~~~~~~~~~~~~
    old_outputs = model_old(inputs)
    new_outputs = model_new(inputs)

    _, old_preds = torch.max(old_outputs.data, 1)
    _, new_preds = torch.max(new_outputs.data, 1)

    old_matrix.append(old_preds.item())
    new_matrix.append(new_preds.item())
    label_matrix.append(labels.item())

    # 結果が違えば保存する(100イテレートまでの)
    if old_preds != new_preds and i <= 100:
        save_to = (
            f"diff_result/{old_preds.item()}_{new_preds.item()}_{labels.item()}.png"
        )
        # これで出来ない可能性大 (次元を減らす)
        # inputs.squeeze()などで
        save_image(inputs, save_to)

    if i % 500 == 0:
        new_accuracy = accuracy_score(label_matrix, new_matrix)
        old_accuracy = accuracy_score(label_matrix, old_matrix)

        print(f"data size: {i+1}, new acc: {new_accuracy}, old acc: {old_accuracy}")

    i += 1


print("creating confusion matrixies")
# confusion matrix作成
for prefix in ["old", "new"]:
    pred = old_matrix if prefix == "old" else new_matrix
    accuracy = accuracy_score(label_matrix, pred)
    model_name = "vgg16" if prefix == "old" else "resnet18"
    cm = confusion_matrix(label_matrix, pred)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title(f"{model_name} model accuracy: {str(accuracy)}")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.clf()
