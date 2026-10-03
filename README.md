# KumaGPT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmmmm25/KumaGPT/blob/main/Kuma_GPT.ipynb)

KumaGPTは、**小規模な日本語GPTモデルをPyTorchで実装し、事前学習から教師ありファインチューニング（SFT）まで行う個人学習プロジェクト**です。

中心となるのは [Kuma_GPT.ipynb](Kuma_GPT.ipynb) です。Transformerの実装、SentencePieceトークナイザーの学習、日本語Wikipediaによる次トークン予測、会話データによるSFT、文章生成、チェックポイント保存までを扱っています。学習済みモデルを試すためのReact / FastAPI製ローカルチャットアプリも付属しています。

ノートブックは作者が実装し、WebアプリはCodexを利用して作成しました。現在は実験段階で、回答の反復や意味の不整合が残っています。

## モデル構造

既存の学習済みモデルを読み込むのではなく、GPT型のdecoder-only Transformerを定義し、ランダム初期化した重みから学習します。

| 項目 | ノートブックの設定 |
| --- | --- |
| トークナイザー | SentencePiece Unigram |
| 語彙サイズ | 8,000 |
| コンテキスト長 | 256トークン |
| Transformer層数 | 10 |
| 埋め込み次元 | 256 |
| Attentionヘッド数 | 4 |
| ヘッドあたりの次元 | 64 |
| FFN中間次元 | 1,024 |
| 活性化関数 | GELU |
| Dropout | 0.1（残差に加算するAttention / FFN出力） |
| 位置表現 | 学習可能な位置埋め込み |
| 正規化 | Pre-LayerNorm、最終LayerNorm |
| 重み共有 | 入力のトークン埋め込みと出力層の重み |
| パラメータ数 | 10,019,648（約10M、上記設定に対応する記録） |

Attentionには `F.scaled_dot_product_attention(..., is_causal=True)` を使用します。未来のトークンを参照せず、直前までの文脈から次のトークンを予測する構造です。

> コード中の `d_head` は全ヘッドを合わせた投影次元（256）です。1ヘッドの次元は `d_head / h = 64` になります。

## 学習の流れ

### 1. データとトークナイザー

事前学習にはHugging Faceの [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) の日本語スナップショット `20231101.ja` を使用します。

- ストリーミングで取得し、seed 42、buffer size 10,000でシャッフル。
- 最初の95,000記事を学習用、次の5,000記事を検証用に使用。
- 学習用本文から、語彙数8,000、character coverage 0.9995のUnigramトークナイザーを学習。
- `<user>`、`<assistant>`、`<|endoftext|>` を独自の特殊トークンとして登録。

トークナイザーの出力は `kumagpt_unigram.model` と `kumagpt_unigram.vocab` です。**推論には、重みの学習時と同じ `.model` が必要**です。同じ語彙数でも、再学習でトークンIDが変われば互換性はありません。

### 2. 事前学習

記事をトークン化し、末尾に `<|endoftext|>` を付けて連結します。これを257トークンずつに分割し、先頭256トークンを入力、1トークンずらした256トークンを正解として学習します。端数は切り捨てます。

```python
input = batch_token[:, :-1]
target = batch_token[:, 1:]
logits, loss = model(input, target)
```

| 項目 | 設定 |
| --- | --- |
| バッチサイズ | 32 |
| 最大入力長 | 256 |
| 通常の1ステップの予測対象数 | 8,192トークン（32 × 256） |
| エポック数 | 20 |
| Optimizer | AdamW |
| 初期学習セルの学習率 | 5e-4 |
| 学習率スケジュール | 500ステップのLinear warmup → Cosine decay |
| 最小学習率 | 1e-5 |
| 損失 | 次トークン予測のCross Entropy |

各エポックで学習・検証lossを計算し、「日本の首都である東京は」に続く文章を生成して、`pretrained{i}.pt` を保存します。ノートブックの記録では、連結した学習データは102,520,333トークンです。この値は保存済み実行の結果で、データ取得やトークナイザーの変更により変わります。

### 3. 会話データによるSFT

[llm-jp/oasst1-21k-ja](https://huggingface.co/datasets/llm-jp/oasst1-21k-ja) の会話を次の形式に変換し、事前学習後のモデルを全パラメータ更新でファインチューニングします。

```text
<user>ユーザーの発言
<assistant>アシスタントの回答<|endoftext|>
```

**入力には会話全体を含め、損失はアシスタントの回答と回答終了トークンだけで計算**します。ユーザー発言、ロールトークン、paddingのラベルは `-1` とし、`ignore_index=-1` で除外します。

```python
input = batch_token[:, :-1]
target = batch_label[:, 1:]
# model.forward内
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    target.reshape(-1),
    ignore_index=-1,
)
```

- データは先頭19,047件を学習用、残りを検証用に分割します。
- 会話ごとに先頭257トークンまで切り詰め、短い会話はpaddingします。
- バッチサイズ32、学習率5e-4、20エポックを設定しています。
- 事前学習とは別のAdamWとwarmup / cosine schedulerを作成します。
- 教師トークンがないバッチ、lossが非有限値のバッチはスキップします。
- 各エポックで3種類の質問への生成を確認し、`sft_model_{i}.pt` を保存します。

会話データ内の複数ターンはテンプレートで処理しますが、長い会話の後半は切り捨てられるため、すべてのターンが学習に使われるわけではありません。

## 記録された結果

以下は、リポジトリ内のノートブックに保存されている実行出力です。今回新たに学習・推論を実行した結果ではありません。セルの設定と保存済み出力は、編集や再実行の順序によって一致しないことがあります。

| 段階 | 出力上のepoch | Train loss | Validation loss |
| --- | --- | --- | --- |
| 事前学習 | 19 | 3.269116 | 3.231879 |
| SFT | 19 | 3.114350 | 2.823942 |
| SFT | 34 | 1.564673 | 1.974916 |
| SFT | 38 | 1.511642 | 1.977618 |

SFTセルの表示は `i + 19`、保存ファイルとチェックポイントのepochは `i` です。「epoch 38」の表示だけから、同じ設定で39エポック連続学習したとは判断できません。

事前学習とSFTでは、データとlossの計算対象が異なるため、lossを単純に性能比較することはできません。また、トークナイザーを変更した実験同士でもlossの尺度は変わります。現在のloss集計はバッチごとの平均で、全教師トークンをまとめた厳密な平均ではありません。

### 生成例

ノートブックの単発推論セルに保存された出力：

```text
<user>寝てもいい? <assistant>もちろん!あなたがいることはありますか?
```

SFTの表示epoch 38に保存された出力の一部：

```text
<user>犬ってどんな動物? <assistant>犬の祖先が犬の祖先だと仮定すると、犬の祖先が犬の祖先に移るか、犬が祖先に移るかどうかは、異なります。
```

回答形式は生成できる一方で、内容の整合性や反復抑制には課題があります。lossの低下だけで自然な会話が実現したとはいえず、固定質問への生成も併せて確認しています。

## ノートブックを動かす

Google ColabのGPUランタイムでの実行を想定しています。メタデータにはA100が記録されていますが、必要なGPUメモリ量や学習時間の比較は未計測です。

1. 上部の「Open In Colab」からノートブックを開き、GPUランタイムを選択します。
2. 必要なライブラリをインストールします。ノートブックのインストールセルは `datasets` のみなので、`sentencepiece` も用意します。
3. Google Driveをマウントし、`save_dir` と `tok_dir` を自分の保存先に変更します。学習前に `os.makedirs(save_dir, exist_ok=True)` で保存先を作成します。
4. Wikipedia取得、トークナイザー学習、モデル定義、トークン化、バッチ作成の各セルを実行します。
5. **初回は「初期学習用」セルを実行し、「学習再開用」セルはスキップ**して事前学習を進めます。
6. 事前学習後のモデルを保持したまま、会話データの読み込み・テンプレート変換・SFTの各セルを実行します。
7. モデル重みと、それに対応するトークナイザーをセットで保存します。

```python
!pip install -q datasets sentencepiece
# PyTorchはColabのランタイムに応じて用意してください。
```

### 再開と保存の注意点

- 再開時は `checkpoint_path` を変更し、学習時と同じトークナイザー・モデル設定・scheduler設定を使用してください。元のトークナイザーを読み込み、再学習しないことが重要です。
- 再開用セルの後に初期学習用セルを実行すると、モデルとOptimizerが新しく作り直されます。どちらか一方を選択してください。
- 事前学習チェックポイントにはモデル・Optimizer・schedulerの状態とconfigを保存します。SFTチェックポイントには `sft_optimizer_state_dict` と `sft_scheduler_state_dict` を保存します。SFT再開専用セルは現在ありません。
- 最終セルの `KumaGPT_2.pt` は現在のモデル重みを保存しますが、Optimizer / schedulerは事前学習側の変数を参照しています。推論には使えますが、SFT再開用には各エポックの `sft_model_{i}.pt` を参照してください。
- リポジトリには学習済み重み、トークナイザー、学習データを同梱していません。
- トークナイザー学習は記事を連結した文字列を入力ファイルに書き込みます。SentencePieceの行長制限によるスキップが発生していないか、学習ログを確認してください。

## 文章生成

生成は自己回帰方式で、直近256トークンを入力に次の1トークンをサンプリングします。既定値はtemperature 0.7、top-k 30です。`<user>` または `<|endoftext|>` を生成すると停止します。

モデルとトークナイザーを読み込んだ状態で、ノートブックから次のように試せます。

```python
prompt = "<user>寝てもいい？\n<assistant>"
x = torch.tensor(
    tokenizer.encode(prompt, out_type=int),
    dtype=torch.long,
).unsqueeze(0).to(device)

y = model.generate(x, 100, temperature=0.7, top_k=30)
print(tokenizer.decode(y[0].tolist()))
```

## リポジトリ構成

```text
KumaGPT/
├── Kuma_GPT.ipynb                 # モデル実装・事前学習・SFT・生成
├── KumaGPT_backend_requirements.md
├── backend/
│   ├── app/                      # FastAPI・DB・推論処理
│   ├── artifacts/                # 重みとトークナイザーの配置先
│   ├── requirements.txt
│   ├── requirements-model.txt
│   └── tests/
└── frontend/
    └── src/                      # Reactのチャット画面
```

## 付属のWebアプリ

学習結果をブラウザーから試すためのローカルアプリです。React / Viteのフロントエンドと、FastAPI / SQLAlchemy / SQLiteのバックエンドをHTTP APIで接続します。保存会話、一時会話、タイトル変更、会話削除に対応します。認証とストリーミングは未対応です。

### 起動（Windows / PowerShell）

Python 3.11または3.12を推奨します。ターミナルを2つ開いて実行してください。

```powershell
# バックエンド
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

```powershell
# 別ターミナルでフロントエンド
cd frontend
Copy-Item .env.example .env
npm.cmd install
npm.cmd run dev
```

画面は http://localhost:5173 、APIドキュメントは http://localhost:8000/docs で開けます。既定の `demo` モードは入力を返す動作確認用で、KumaGPTの推論結果ではありません。

### 学習済みモデルを接続する

1. `KumaGPT_2.pt` を `backend/artifacts/KumaGPT_2.pt` に配置します。
2. 同じ学習で使用した `kumagpt_unigram.model` を `backend/artifacts/tokenizer.model` という名前で配置します（ファイル名の変更のみ）。
3. バックエンドの仮想環境で `python -m pip install -r requirements-model.txt` を実行します。
4. `backend/.env` を変更し、バックエンドを再起動します。

```dotenv
KUMAGPT_MODEL_BACKEND=torch
KUMAGPT_MODEL_PATH=./artifacts/KumaGPT_2.pt
KUMAGPT_TOKENIZER_PATH=./artifacts/tokenizer.model
```

推論時はCUDAが利用可能ならGPU、そうでなければCPUを使用します。Webアプリは直近10件のメッセージからプロンプトを作成しますが、モデルに実際に渡るのは最大256トークンです。

APIテスト：

```powershell
cd backend
python -m pytest -q
```

## 現在の課題

- 生成文の反復、事実誤認、質問と回答の意味のずれ。
- 256トークンの文脈制限と、SFTでの先頭切り詰めによる回答・後続ターンの欠落。
- 固定質問以外を含む評価セットと、実験条件・チェックポイントの一貫した記録。
- SFTの再開処理と、最終エクスポートの学習状態メタデータの整理。

データを利用・再配布する際は、各データセットのライセンスと利用条件を確認してください。
