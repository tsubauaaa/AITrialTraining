### 認証機能をつけたサイコロ認識API
画像をPostするとその画像内にあるサイコロを検出する画像を返すAPI。実行はlocal、Dockerどちらでも可能。リクエストには認証トークンが必要
EC2で実行してアクセスして確認する

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
$ curl -X POST "http://servername:8000/predict" -F file=@./static/images/upload/dice-example4.jpg -H 'Authorization: Bearer token' -o ./result.jpg
```
※ `servername`は実行環境がlocalかクラウドかで異なります

#### 不要なDockerコンポーネントを削除
```
$ docker system prune -f
```

#### デモ
このAPIをEC2(ec2-52-195-9-43.ap-northeast-1.compute.amazonaws.com)上で実行して、サイコロ画像をPOSTしてラベルと検出境界線が書かれた画像を取得するデモ

![Demo](https://github.com/tsubauaaa/AITrialTraining/blob/main/Training6/dice_api/demo.gif)