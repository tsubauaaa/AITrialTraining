import copy
import pickle
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


def train_model(
    model, criterion, optimizer, dataloaders, scheduler=None, num_epochs=25
):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    model = model.to(device)

    # 始まりの時間
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 途中経過保存用に、リストを持った辞書を作ります。
    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    for epoch in tqdm(range(num_epochs)):
        if (epoch + 1) % 5 == 0:  # ５回に１回エポックを表示します。
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

        # それぞれのエポックで、train, valを実行します。
        # 辞書に入れた威力がここで発揮され、trainもvalも１回で書く事ができます。
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 学習モード。dropoutなどを行う。
            else:
                model.eval()  # 推論モード。dropoutなどを行わない。

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data  # ImageFolderで作成したデータは、データをラベルを持ってくれます。

                # GPUを使わない場合不要
                if use_gpu:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                # ~~~~~~~~~~~~~~forward~~~~~~~~~~~~~~~
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                # torch.maxは実際の値とインデクスを返します。
                # torch.max((0.8, 0.1),1)=> (0.8, 0)
                # 引数の1は行方向、列方向、どちらの最大値を返すか、です。
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics #GPUなしの場合item()不要
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                # (preds == labels)は[True, True, False]などをかえしますが、
                # pythonのTrue, Falseはそれぞれ1, 0に対応しているので、
                # sumで合計する事ができます。

            # サンプル数で割って平均を求めます。
            # 辞書にサンプル数を入れたのが生きてきます。
            epoch_loss = running_loss / data_size[phase]
            # GPUなしの場合item()不要
            epoch_acc = running_corrects.item() / data_size[phase]

            # リストに途中経過を格納
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            # tensot().item()を使う事で、テンソルから値を取り出す事ができます。
            # print(tensorA)       => tensor(112, device='cuda:0')
            # print(tensorA.itme)) => 112

            # formatを使いますが、.nfとすると、小数点以下をn桁まで出力できます。
            # C言語と一緒ですね。
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # 精度が改善したらモデルを保存する
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # deepcopyをしないと、model.state_dict()の中身の変更に伴い、
            # コピーした（はずの）データも変わってしまいます。
            # copyとdeepcopyの違いはこの記事がわかりやすいです。
            # https://www.headboost.jp/python-copy-deepcopy/

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val acc: {:.4f}".format(best_acc))

    # 最良のウェイトを読み込んで、返す。
    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict


if __name__ == "__main__":
    transform_dict = {
        "train": transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }
    data_folder = "raw-img"

    print("loading data")
    data = torchvision.datasets.ImageFolder(
        root=data_folder, transform=transform_dict["train"]
    )

    train_ratio = 0.8
    train_size = int(train_ratio * len(data))

    val_size = len(data) - train_size
    data_size = {"train": train_size, "val": val_size}
    data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])

    batch_size = 16

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    # model 1
    model_new = models.resnet18(pretrained=True)
    # 今回は10クラスで、組み込みのモデルは1000クラス？くらいなので書き換える
    num__new_model_features = model_new.fc.in_features
    model_new.fc = nn.Linear(num__new_model_features, 10)

    # model 2
    model_old = models.vgg16(pretrained=True)
    model_old.classifier[6] = nn.Linear(in_features=4096, out_features=10)

    print(model_new)
    print(model_old)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    lr = 1e-4
    epoch = 10
    old_optim = torch.optim.Adam(model_old.parameters(), lr=lr, weight_decay=1e-4)
    new_optim = torch.optim.Adam(model_new.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # oldバージョンの学習
    print("start training old model")
    model_old_ft, loss, acc = train_model(
        model_old, criterion, old_optim, dataloaders, num_epochs=epoch
    )

    # 学習済みモデルの保存
    torch.save(model_old_ft.state_dict(), "model_old_ft.pt")

    # newバージョンの学習
    print("start training new model")
    model_new_ft, loss, acc = train_model(
        model_new, criterion, new_optim, dataloaders, num_epochs=epoch
    )

    # 学習済みモデルの保存
    torch.save(model_new_ft.state_dict(), "model_new_ft.pt")

    with open("data_train.pkl", "wb") as f:
        pickle.dump(data_train, f)

    with open("data_val.pkl", "wb") as f:
        pickle.dump(data_val, f)
