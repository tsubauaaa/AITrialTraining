### Fargateでテキスト要約APIをデプロイ(API Gateway + Cognito)
テキストをリクエストすると要約したテキストを返すAPI。実行はlocal、Dockerどちらでも可能。
FargateにデプロイしてAPI GatewayのAuthorizerを付けて、アクセスして確認する

#### localで実行する場合
```
$ pip install -r requirements.txt
$ python main.app
```

#### dockerで実行する場合
```
$ docker-compose up -d --build
```

#### api requestをcurlで試す場合
```
$ curl -X POST -H "Content-Type: application/json" -d '{"review_body": "Pathetic design of the caps. Very impractical to use everyday. The caps close so tight that everyday we have to wrestle with the bottle to open the cap. With a baby in one hand opening the cap is a night mare. And on top of these extra ordinary features of super secure cap, they are so expensive when compared to other brands. Stay away from these until they fix the cap issues. We have hurt ourselves many time trying to open caps as they have sharp edges on the inner and outer edges. Not worth the price."}' servername:8001/predict
```
※ `servername`は実行環境がlocalかクラウドかで異なります

#### 不要なDockerコンポーネントを削除
```
$ docker system prune -f
```

#### デモ
