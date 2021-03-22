## 課題6
Googleのモデルを盗んでみる  
認証機能をつけたサイコロ認識API

### [google/](./google/)
Googleのモデルを盗んでみる  
Vision APIのオブジェクト検出APIでKaggleのAnimal10画像の分類を行い、その分類結果からAnimal10のVision APIバージョンのラベルを作成


### [dice_api/](./dice_api/)
認証機能をつけたサイコロ認識API
画像をPostするとその画像内にあるサイコロを検出する画像を返すAPI。実行はlocal、Dockerどちらでも可能。リクエストには認証トークンが必要