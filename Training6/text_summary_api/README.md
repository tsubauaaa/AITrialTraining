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

#### Fargateデプロイ
Fargateにテキスト要約APIをデプロイする際の構成やAPI GatewayのAuthorizerの構成などを記載する  
また、Fargate上のAPIにアクセスして確認する

##### Authorizerのトークン取得する
```
$ aws cognito-idp admin-initiate-auth --user-pool-id ap-northeast-1_AbNe5La9Z --client-id 3oco5uqm9iatihv56c60d805om --auth-flow ADMIN_NO_SRP_AUTH --auth-parameters USERNAME=ts-hirota,PASSWORD=******
```
レスポンス
```
{
    "ChallengeParameters": {},
    "AuthenticationResult": {
        "AccessToken": "**********",
        "ExpiresIn": 3600,
        "TokenType": "Bearer",
        "RefreshToken": "**********",
        "IdToken": "**********"
    }
}
```
上記のIdTokenをAPI GatewayのAuthorizerで使う

##### API Gateway -> NLB -> Fargateにリクエストする
```
$ curl -X POST -H "Content-Type: application/json" -H 'Authorization: Bearer "**********" -d '{"review_body": "Pathetic design of the caps. Very impractical to use everyday. The caps close so tight that everyday we have to wrestle with the bottle to open the cap. With a baby in one hand opening the cap is a night mare. And on top of these extra ordinary features of super secure cap, they are so expensive when compared to other brands. Stay away from these until they fix the cap issues. We have hurt ourselves many time trying to open caps as they have sharp edges on the inner and outer edges. Not worth the price."}' https://2r61vfii5l.execute-api.ap-northeast-1.amazonaws.com/dev/
```
レスポンス
```
"As a mother of a newborn baby, I have to say that the bottle caps on some of the baby bottles are a bit of a let down."
```

#### デモ
このAPIをFargate上で実行して、API Gateway(https://2r61vfii5l.execute-api.ap-northeast-1.amazonaws.com/dev/)経由でアクセスする  
デモでは、最初にAuthorizationヘッダーなしでアクセスする  
すると以下のように認証エラーが帰ってくる
```
{"message":"Unauthorized"}%
```
次にAuthorizationヘッダーを付けてアクセスして、要約結果を受け取る

![Demo](https://github.com/tsubauaaa/AITrialTraining/blob/main/Training6/text_summary_api/demo.gif)
