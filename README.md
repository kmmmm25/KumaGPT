# KumaGPT

要件定義書をもとにした、ローカル専用チャットアプリです。フロントエンド（React）とバックエンド（FastAPI + SQLAlchemy）を分離しています。

## 起動方法

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

`http://localhost:8000/docs`でAPIを試せます。既定の`demo`モードはモデルなしで画面とDB操作を確認できます。

```powershell
# 別ターミナルでフロントエンド
cd frontend
Copy-Item .env.example .env
npm.cmd install
npm.cmd run dev
```

ブラウザーで`http://localhost:5173`を開きます。

## 未完成のKumaGPTを接続するために必要なもの

Notebookの最終セルが保存する`KumaGPT_2.pt`だけでなく、学習時と同じSentencePieceモデルも必要です。

1. Notebookで`KumaGPT_2.pt`を保存する。
2. トークナイザー学習セルで作った`tokenizer.model`も保存する（`.vocab`だけでは推論できません）。
3. `backend/artifacts/KumaGPT_2.pt`と`backend/artifacts/tokenizer.model`へ配置する。
4. `backend/.env`を次のように変更し、バックエンドを再起動する。

```dotenv
KUMAGPT_MODEL_BACKEND=torch
KUMAGPT_MODEL_PATH=./artifacts/KumaGPT_2.pt
KUMAGPT_TOKENIZER_PATH=./artifacts/tokenizer.model
```

実モデルを使う環境では、通常の依存関係に加えて次を実行してください。

```powershell
python -m pip install -r requirements-model.txt
```

Notebook側では、トークナイザー作成直後に次を実行して`.model`をGoogle Drive等へ退避してください。

```python
import shutil
shutil.copy("tokenizer.model", "/content/drive/MyDrive/KumaGPT/check/tokenizer.model")
```

チェックポイントには`config`と`model_state_dict`が必要です。現在のNotebook最終セルは両方を保存しています。必ず`KumaGPT_2.pt`を作成した時点と同じ`tokenizer.model`を組にしてください。語彙サイズやIDが違うとロードまたは生成に失敗します。

## 動作確認

```powershell
cd backend
python -m pytest -q
```

保存会話の作成・履歴・タイトル変更・削除、一時会話のDB非保存、直近10件の文脈、推論失敗時の非保存に対応しています。一時会話はReactのメモリだけにあり、再読み込みで消えます。認証、ストリーミング、外部公開は未対応です。
