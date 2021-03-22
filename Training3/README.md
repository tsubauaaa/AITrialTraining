## 課題3
Amazonレビュー(日本語)をMeCabで形態素解析して、word2vecでベクトル化し、LSTMで評価の星の数を回帰分析する

### [AITraining3.ipynb](./AITraining3.ipynb)
Amazonレビュー(日本語)をMeCabで形態素解析して、word2vecでベクトル化し、LSTMで評価の星の数を回帰分析する
回帰結果を四捨五入して、Ground Truthとマッチしている数から精度を算出

### 推論結果
```
predict : 0.35329795299469297, match: 466, tested_num: 1319
```
テストケース1319のうち正解数466の精度35%だった