# Frontend AI

ブラウザだけで動作する小型AIです。

## 特徴

- JavaScriptだけで動作
- サーバー側AI API不要
- JSON学習データ
- 7種類の学習データ
- ブラウザ上で学習
- 学習済みモデルをlocalStorageに保存
- 会話履歴をlocalStorageに保存
- Temperature
- Top-K
- Top-P
- 最大生成文字数
- 学習Epoch変更
- JSON学習データ追加
- JSON学習データ出力
- JSON学習データ読み込み
- モデルリセット

## フォルダ

frontend-ai/

├── index.html

├── README.md

├── css/

│   └── style.css

├── js/

│   ├── app.js

│   ├── config.js

│   ├── tokenizer.js

│   ├── model.js

│   ├── trainer.js

│   ├── storage.js

│   ├── generator.js

│   └── data-loader.js

└── data/

    ├── training-01.json

    ├── training-02.json

    ├── training-03.json

    ├── training-04.json

    ├── training-05.json

    ├── training-06.json

    └── training-07.json

## 起動方法

JSONファイルをfetchで読み込むため、index.htmlを直接ダブルクリックしないでください。

Pythonがインストールされている場合は、frontend-aiフォルダをターミナルで開きます。

以下を実行します。

python -m http.server 8000

その後、ブラウザで以下を開きます。

http://localhost:8000/

## 学習

画面下の「学習」を押します。

設定からEpochを変更できます。

最初は30Epoch程度がおすすめです。

## 学習データ追加

「＋ 学習データ」を押します。

入力と出力を入力して追加できます。

追加したデータはブラウザのlocalStorageに保存されます。

その後「学習」を押してください。

## JSON出力

「JSON出力」を押すと現在の学習データをJSONとして保存できます。

## JSON読込

「JSON読込」から独自の学習データJSONを読み込めます。

形式は以下です。

[
  {
    "input": "こんにちは",
    "output": "こんにちは！"
  },
  {
    "input": "AIとは？",
    "output": "人工知能のことです。"
  }
]

## 注意

このプロジェクトは教育用・小型AI向けです。

ChatGPTなどの大規模言語モデルと同じ性能ではありません。

現在のモデルは小型のニューラルネットワークであり、本格的なTransformer LLMではありません。

さらに性能を上げる場合は、

- Web Worker
- IndexedDB
- Subword Tokenizer
- Attention
- Transformer
- Mini-batch
- Adam Optimizer
- 学習データ増加
- 文脈メモリ
- モデル量子化

などを追加できます。
