## 課題2
[課題1](../Training1)で実施したyolov5を用いたサイコロのオブジェクト検出学習にmlflowを実装して、学習パラメータを変えた場合の管理を行う
また、パラメータ引数をhydra管理にして、multirunなどを行う
それぞれの学習パラメータを変えた学習をmlflow uiにて確認する

### [train_mlflow_hydra.py](./train_mlflow_hydra.py)
yolov5のtrain.pyにmlflowとhydraを実装したもの
DataAugmentationのdegreesは (1, 0.0, 45.0)

### [train_mlflow_hydra_da_change.py](./train_mlflow_hydra_da_change.py)
yolov5のtrain.pyにmlflowとhydraを実装したもの
DataAugmentationのdegreesは (10, 5.0, 0.0)に変更

### [config/config.yaml](./config/config.yaml)
[train_mlflow_hydra.py](./train_mlflow_hydra.py)と[train_mlflow_hydra_da_change.py](./train_mlflow_hydra_da_change.py)が読み込む学習パラメータファイル

### [AITraining2.ipynb](./AITraining2.ipynb)
[train_mlflow_hydra.py](./train_mlflow_hydra.py)と[train_mlflow_hydra_da_change.py](./train_mlflow_hydra_da_change.py)を使って、パラメータが、

* DataAugmentation degrees: (1, 0.0, 45.0), epochs: 5, batch-size: 16
* DataAugmentation degrees: (10, 5.0, 0.0), epochs: 5, batch-size: 16
* DataAugmentation degrees: (1, 0.0, 45.0), epochs: 5, batch-size: 8, 16 (multirun)
* DataAugmentation degrees: (1, 0.0, 45.0), epochs: 5, 10, batch-size: 16 (multirun)  

の場合の合計6回、5パターンを実行する

### [multirun](./multirun)
mlflow multirun実行時のパラメータ

### mlflow ui結果

| 学習パラメータ | mlflow ui |
| ---- | ---- |
| degrees: (1, 0.0, 45.0), epochs: 5, batch-size: 16 | [mlflow ui pdf1](./report_materials/1.pdf) |
| degrees: (10, 5.0, 0.0), epochs: 5, batch-size: 16 | [mlflow ui pdf2](./report_materials/2.pdf) |
| degrees: (1, 0.0, 45.0), epochs: 5, batch-size: 8 | [mlflow ui pdf3](./report_materials/3.pdf) |
| degrees: (1, 0.0, 45.0), epochs: 5, batch-size: 16 | [mlflow ui pdf4](./report_materials/4.pdf) |
| degrees: (1, 0.0, 45.0), epochs: 10, batch-size: 16 | [mlflow ui pdf5](./report_materials/5.pdf) |