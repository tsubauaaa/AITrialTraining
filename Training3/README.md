## 課題3
Amazonレビュー(日本語)をMeCabで形態素解析して、word2vecでベクトル化し、LSTMで評価の星の数を回帰分析する

### [AITraining3.ipynb](./AITraining3.ipynb)
Amazonレビュー(日本語)をMeCabで形態素解析して、word2vecでベクトル化し、LSTMで評価の星の数を回帰分析する
正答率を回帰結果を四捨五入して、Ground Truthとマッチしている数から精度を算出
回帰結果とGround Truthとの平均絶対誤差も求める
### 推論結果
```
Correct answer rate: 0.358683314415437, match: 316, tested_num: 881
Mean Absolute Error: 0.9054456292382028
```
テストケース881のうち正答数316の正答率36%ほどだった  
回帰結果とGround Truthとの平均絶対誤差は0.9ほどだった
バッチとパディングを対応すると正答率33->36%、平均絶対誤差1.3->0.9とどちらも良くなった

※ 実際のプログラム実施は[AITraining3.ipynb](./AITraining3.ipynb)を参照