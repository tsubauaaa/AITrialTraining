## 課題6
Googleのモデルを盗んでみる  
EC2でサイコロ認識APIをデプロイ
Fargateでテキスト要約APIをデプロイ(API Gateway + Cognito)

### [google/](./google/)
Googleのモデルを盗んでみる  
Vision APIのオブジェクト検出APIでKaggleのAnimal10画像の分類を行い、その分類結果からAnimal10のVision APIバージョンのラベルを作成


### [dice_api/](./dice_api/)
EC2でサイコロ認識APIをデプロイ
画像をPostするとその画像内にあるサイコロを検出する画像を返すAPI。実行はlocal、Dockerどちらでも可能。リクエストには認証トークンが必要
EC2にデプロイしてアクセスして確認する


### [text_summary_api/](./text_summary_api/)
Fargateでテキスト要約APIをデプロイ(API Gateway + Cognito)
テキストをリクエストすると要約したテキストを返すAPI。実行はlocal、Dockerどちらでも可能。
FargateにデプロイしてAPI GatewayのAuthorizerを付けて、アクセスして確認する
