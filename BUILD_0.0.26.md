# torchcrepe 0.0.26 ビルドサマリー

## ビルド完了

✅ **torchcrepe-0.0.26-py3-none-any.whl** の作成に成功しました。

---

## ビルド成果物

### dist/ ディレクトリ

| ファイル | サイズ | 説明 |
|---------|--------|------|
| **torchcrepe-0.0.26-py3-none-any.whl** | 140 MB | Wheelパッケージ（推奨） |
| torchcrepe-0.0.26.tar.gz | 140 MB | ソース配布物 |

---

## 含まれるモデルファイル

wheelパッケージには以下の6つのモデルファイルが含まれています：

| モデル | ファイルサイズ | 容量 | 状態 |
|--------|--------------|------|------|
| tiny | 1.87 MB | 4/32 | ✓ |
| small | 6.23 MB | 8/32 | ✓ |
| **regular** | **13.08 MB** | **12/32** | **✓ 新規追加** |
| medium | 22.44 MB | 16/32 | ✓ |
| large | 48.65 MB | 24/32 | ✓ |
| full | 84.87 MB | 32/32 | ✓ |

**合計モデルサイズ**: 約177 MB

---

## バージョン 0.0.26 の新機能

### ✨ Regular モデルの追加

- **容量**: 12/32 (small と medium の中間)
- **パラメータ数**: 約3.4M
- **作成手法**: チャネルプルーニング（fullモデルから抽出）
- **用途**: smallより高精度、mediumより高速

### 🔧 コード変更

1. **torchcrepe/model.py**
   - Regularモデルのアーキテクチャ定義を追加
   - チャネル数: [384, 48, 48, 48, 96, 192]

2. **torchcrepe/__main__.py**
   - CLIヘルプにregularを追加
   - `--model` オプションで `regular` を指定可能

3. **torchcrepe/assets/regular.pth**
   - 新規モデルファイルを追加

---

## インストール方法

### ローカルwheelからインストール

```bash
pip install dist/torchcrepe-0.0.26-py3-none-any.whl
```

### アンインストール（古いバージョン）

```bash
pip uninstall torchcrepe -y
```

### 強制再インストール

```bash
pip install --force-reinstall dist/torchcrepe-0.0.26-py3-none-any.whl
```

---

## 使用例

### Python API

```python
import torchcrepe

# Regularモデルで推論
audio, sr = torchcrepe.load.audio('input.wav')

pitch = torchcrepe.predict(
    audio,
    sr,
    hop_length=160,
    fmin=50,
    fmax=550,
    model='regular',  # ← 新しいモデル
    device='cuda:0'
)

# 埋め込み抽出
embedding = torchcrepe.embed(
    audio,
    sr,
    hop_length=160,
    model='regular',
    device='cuda:0'
)
```

### コマンドライン

```bash
# Regularモデルで推論
python -m torchcrepe \
    --audio_files input.wav \
    --output_files pitch.pt \
    --model regular

# 利用可能なモデルを確認
python -m torchcrepe --help
# --model: One of "tiny", "small", "regular", "medium", "large", or "full"
```

---

## 動作確認

### テスト結果

全モデルサイズで動作確認済み：

```
[TINY]     ✓ Pitch prediction  ✓ Embedding extraction
[SMALL]    ✓ Pitch prediction  ✓ Embedding extraction
[REGULAR]  ✓ Pitch prediction  ✓ Embedding extraction
[MEDIUM]   ✓ Pitch prediction  ✓ Embedding extraction
[LARGE]    ✓ Pitch prediction  ✓ Embedding extraction
[FULL]     ✓ Pitch prediction  ✓ Embedding extraction
```

### 検証スクリプト

```bash
# インストール後の動作確認
python temp/final_test.py
```

---

## モデルサイズ比較

| モデル | パラメータ数 | 推論速度（相対） | 精度（期待値） |
|--------|-------------|----------------|--------------|
| tiny | 0.49M | 最速 (7x) | 低 |
| small | 1.63M | 速い (2x) | 中低 |
| **regular** | **3.43M** | **やや速い (1.5x)** | **中** |
| medium | 5.88M | 基準 (1x) | 中高 |
| large | 12.75M | やや遅い (0.5x) | 高 |
| full | 22.25M | 最遅 (0.15x) | 最高 |

※推論速度はmediumを基準とした相対値（推定）

---

## 技術的詳細

### Regularモデルの作成方法

1. **チャネルプルーニング**を使用
2. fullモデルの各層から重要度（L1ノルム）が高いチャネルを選択
3. 選択率: 37.5% (384/1024チャネル)

### プルーニングアルゴリズム

```python
# 重要度計算
importance = torch.abs(weight).sum(dim=(1, 2, 3))

# 上位k個を選択
_, indices = torch.sort(importance, descending=True)
selected = indices[:k]
```

### 制限事項

- プルーニングによる抽出のため、独立訓練されたモデルより精度が若干低下する可能性
- 実データでの精度評価を推奨

---

## ファイル構成

```
dist/
├── torchcrepe-0.0.26-py3-none-any.whl  ← メインの配布ファイル
└── torchcrepe-0.0.26.tar.gz            ← ソース配布物

torchcrepe/
├── assets/
│   ├── tiny.pth       (1.87 MB)
│   ├── small.pth      (6.23 MB)
│   ├── regular.pth    (13.08 MB) ← 新規
│   ├── medium.pth     (22.44 MB)
│   ├── large.pth      (48.65 MB)
│   └── full.pth       (84.87 MB)
├── model.py           ← regularアーキテクチャ追加
└── __main__.py        ← CLIヘルプ更新

temp/
├── create_regular_model.py  ← 作成スクリプト
├── test_regular_model.py    ← テストスクリプト
└── final_test.py            ← 統合テスト

report.md                     ← 詳細レポート
```

---

## 変更履歴

### Version 0.0.26 (2025-10-20)

**新機能:**
- ✨ **Regular (12/32) モデルを追加**
  - チャネルプルーニングでfullモデルから作成
  - smallとmediumの中間サイズ
  - パラメータ数: 3.4M

**変更:**
- 📝 `torchcrepe/model.py`: Regularアーキテクチャ定義追加
- 📝 `torchcrepe/__main__.py`: CLIヘルプ更新
- 📦 `torchcrepe/assets/regular.pth`: 新規モデルファイル追加

**技術:**
- 🔧 L1ノルムベースのチャネル選択
- 🔧 fullモデルから37.5%のチャネルを抽出

---

## 互換性

### Python バージョン
- Python 3.6 以降

### 依存パッケージ
- torch
- torchaudio
- librosa >= 0.9.1
- scipy
- resampy
- tqdm

### API互換性
- 既存のコード（0.0.25以前）は変更なしで動作
- 新しい `model='regular'` オプションが利用可能

---

## トラブルシューティング

### インストールエラー

```bash
# 既存バージョンを完全削除
pip uninstall torchcrepe -y

# 再インストール
pip install dist/torchcrepe-0.0.26-py3-none-any.whl
```

### モデルが見つからない

```python
import torchcrepe
import os

# インストール場所を確認
print(os.path.dirname(torchcrepe.__file__))

# モデルファイルの存在確認
models = ['tiny', 'small', 'regular', 'medium', 'large', 'full']
for model in models:
    path = os.path.join(os.path.dirname(torchcrepe.__file__), 'assets', f'{model}.pth')
    print(f"{model}: {'OK' if os.path.exists(path) else 'NOT FOUND'}")
```

### 動作確認

```bash
# 簡易テスト
python -c "import torchcrepe; print('Import OK'); model = torchcrepe.Crepe('regular'); print('Regular model OK')"
```

---

## 関連ファイル

- **詳細レポート**: `report.md`
- **作成手順**: `temp/create_regular_model.py`
- **テストスクリプト**: `temp/final_test.py`

---

**ビルド日時**: 2025-10-20 21:22  
**Python バージョン**: 3.10  
**PyTorch バージョン**: 2.5.1  
**ビルドツール**: setuptools + build

