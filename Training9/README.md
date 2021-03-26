## 課題9
マリオを深層学習でクリアしてみよう  
[Train a MARIO-Playing RL Agent日本語版](https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/4_RL/4_2_mario_rl_tutorial_jp.ipynb)を実行してマリオの面クリアを目指す

### [4_2_mario_rl_tutorial_jp_sagemaker.ipynb](./4_2_mario_rl_tutorial_jp_sagemaker.ipynb)
[Train a MARIO-Playing RL Agent日本語版](https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/4_RL/4_2_mario_rl_tutorial_jp.ipynb)をSageMaker Notebookインスタンスで動くようにもろもろ修正したもの

### マリオ学習結果

| 学習方法 | マリオ結果 | コメント |
| ---- | ---- | ---- |
| [Train a MARIO-Playing RL Agent日本語版](https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/4_RL/4_2_mario_rl_tutorial_jp.ipynb)をepisodes=10000でSageMaker Notebookインスタンス上で[学習](4_2_mario_rl_tutorial_jp_sagemaker.ipynb) | ![Demo](https://github.com/tsubauaaa/AITrialTraining/blob/main/Training9/demo-sagemaker.gif) | episodes=10000でも思うように上手くプレイできなかった。コマンド追加やexploration_rateを変えるともう少しマリオが進むかもしれない |
| [Train a MARIO-Playing RL Agent日本語版](https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/4_RL/4_2_mario_rl_tutorial_jp.ipynb)の原著者の方の[Github](https://github.com/YuansongFeng/MadMario)から学習済みモデルをロードしてepisodes=100で学習 | ![Demo](https://github.com/tsubauaaa/AITrialTraining/blob/main/Training9/demo-clear.gif) | episodes=60辺りからクリアできた |