## 課題7
SageMakerにテキスト要約の独自モデル(transformersのAutoModelForSeq2SeqLM)をデプロイして、リクエストする。


### [source_dir/entry_point.py](./source_dir/entry_point.py)
transformersのAutoModelForSeq2SeqLMを用いて、リクエストされたAmazonレビューテキスト(英語)を要約して返却する。SageMaker推論エンドポイントで実行されるように必要な関数が書かれている

### [source_dir/test.py](./source_dir/test.py)
source_dir/entry_point.pyのテストコード

### [source_dir/args.json](./source_dir/args.json)
transformersのAutoModelForSeq2SeqLMが使う各種パラメータファイル。source_dir/entry_point.pyが使用する

### [inference_test.py](./inference_test.py)
作成したSageMakerエンドポイントにリクエストする

### [AITraining7.ipynb](./AITraining7.ipynb)
SageMaker PyTorchModelを作成し、推論インスタンスをデプロイする。その後、inference_test.pyを実行して、推論をテストする

### 推論結果 (例)
|  Text (Request)  |  Summary (Response)  |
| ---- | ---- |
| Pathetic design of the caps. Very impractical to use everyday. The caps close so tight that everyday we have to wrestle with the bottle to open the cap. With a baby in one hand opening the cap is a night mare. And on top of these extra ordinary features of super secure cap, they are so expensive when compared to other brands. Stay away from these until they fix the cap issues. We have hurt ourselves many time trying to open caps as they have sharp edges on the inner and outer edges. Not worth the price. | As a mother of a newborn baby, I have to say that the bottle caps on some of the baby bottles are a bit of a let down |
実際のリクエスト実施はAITraining7.ipynbを参照