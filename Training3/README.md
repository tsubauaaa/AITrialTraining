## 課題3
Amazonレビュー(日本語)をMeCabで形態素解析して、word2vecでベクトル化し、LSTMで評価の星の数を回帰分析する

### [AITraining3.ipynb](./AITraining3.ipynb)
Amazonレビュー(日本語)をMeCabで形態素解析して、word2vecでベクトル化し、LSTMで評価の星の数を回帰分析する
正答率を回帰結果を四捨五入して、Ground Truthとマッチしている数から精度を算出
回帰結果とGround Truthとの平均絶対誤差も求める
### 推論結果
```
Correct answer rate: 0.3368660105980318, match: 445, tested_num: 1321
Mean Absolute Error: 1.2965448602503848
```
テストケース1321のうち正答数445の正答率33%ほどだった  
回帰結果とGround Truthとの平均絶対誤差は1.3ほどだった

※ 実際のプログラム実施は[AITraining3.ipynb](./AITraining3.ipynb)を参照