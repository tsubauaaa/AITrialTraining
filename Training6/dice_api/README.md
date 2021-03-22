### 認証機能をつけたサイコロ認識API
画像をPostするとその画像内にあるサイコロを検出する画像を返すAPI。実行はlocal、Dockerどちらでも可能。リクエストには認証トークンが必要

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