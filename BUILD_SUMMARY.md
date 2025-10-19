# torchcrepe 拡張版 ビルドサマリー

## 実施内容

torchcrepeフォークに small, medium, large モデルを追加し、wheelパッケージ化を完了しました。

---

## ✅ 完了した作業

### 1. モデル重みファイルの変換

**元のファイル取得:**
- model-small.h5.bz2 (4.2MB) ✓
- model-medium.h5.bz2 (15MB) ✓
- model-large.h5.bz2 (32MB) ✓

**変換結果:**
- small.pth (6.23MB) ✓
- medium.pth (22.44MB) ✓
- large.pth (48.65MB) ✓

変換スクリプト: `temp/convert_h5_to_pth.py`

### 2. コード修正

**torchcrepe/model.py:**
- Crepe.__init__() に small/medium/large の定義を追加
- チャネル数とin_features を正しく設定

**torchcrepe/__main__.py:**
- ヘルプテキストを更新（"tiny", "small", "medium", "large", "full"）

**README.md:**
- モデルサイズの説明を更新

**setup.py:**
- バージョンを 0.0.24 → 0.0.25 に更新

### 3. パッケージング

**ビルド成果物:**
- `dist/torchcrepe-0.0.25-py3-none-any.whl` (130MB)
- `dist/torchcrepe-0.0.25.tar.gz` (130MB)

**含まれるモデルファイル:**
```
torchcrepe/assets/
├── tiny.pth    (1.87 MB)  - 元から存在
├── small.pth   (6.23 MB)  - 新規追加 ✓
├── medium.pth  (22.44 MB) - 新規追加 ✓
├── large.pth   (48.65 MB) - 新規追加 ✓
└── full.pth    (84.87 MB) - 元から存在
```

---

## 📦 インストール方法

### ローカルwheelからインストール

```bash
pip install dist/torchcrepe-0.0.25-py3-none-any.whl
```

### 開発モードでインストール

```bash
pip install -e .
```

---

## 🧪 動作確認済み

### テスト結果

全てのモデルサイズで以下を確認:
- ✅ モデルのインスタンス化
- ✅ 推論実行 (入力: [batch, 1024] → 出力: [batch, 360])
- ✅ エンベディング抽出
- ✅ 重みファイルの読み込み

### 使用例

```python
import torchcrepe

# 全モデルサイズが使用可能
model_sizes = ['tiny', 'small', 'medium', 'large', 'full']

# 例: mediumモデルで推論
audio, sr = torchcrepe.load.audio('input.wav')
pitch = torchcrepe.predict(
    audio, sr,
    model='medium',  # ← 新しいモデルサイズ
    device='cuda:0'
)
```

### CLIの使用例

```bash
# smallモデルで実行
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --model small

# mediumモデルで実行
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --model medium

# largeモデルで実行
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --model large
```

---

## 📊 モデルサイズ比較

| モデル | パラメータ倍率 | ファイルサイズ | 精度 | 速度 |
|--------|---------------|--------------|------|------|
| tiny   | 1/8 (4)       | 1.87 MB      | 低   | 最速 |
| small  | 2/8 (8)       | 6.23 MB      | 中低 | 速い |
| medium | 4/8 (16)      | 22.44 MB     | 中   | 普通 |
| large  | 6/8 (24)      | 48.65 MB     | 中高 | 遅い |
| full   | 8/8 (32)      | 84.87 MB     | 高   | 最遅 |

---

## 🔧 技術的な詳細

### モデルアーキテクチャ

各モデルのチャネル数（6層のCNN）:

```
層          tiny   small  medium large  full
conv1:      128    256    512    768    1024
conv2-4:    16/32  32/64  64/128 96/192 128/256
conv5:      32     64     128    192    256
conv6:      64     128    256    384    512
classifier: 256    512    1024   1536   2048 → 360
```

### 変換プロセス

1. Keras .h5 ファイルをダウンロード
2. h5pyで重みを読み込み
3. 以下の変換を適用:
   - Conv2D: (H,W,C_in,C_out) → (C_out,C_in,H,W)
   - BatchNorm: gamma/beta → weight/bias, moving_mean/variance → running_mean/var
   - Dense: (in,out) → (out,in)
4. PyTorch state_dictとして保存

---

## 📁 作業ファイル

### 重要なスクリプト

```
temp/
├── convert_h5_to_pth.py       - Keras→PyTorch変換スクリプト
├── inspect_h5.py              - h5ファイル構造確認
├── inspect_weights.py         - 重み詳細確認
├── test_installation.py       - インストール検証
└── model-*.h5                 - 元のKeras重みファイル
```

### ビルド成果物

```
dist/
├── torchcrepe-0.0.25-py3-none-any.whl  - Wheelパッケージ
└── torchcrepe-0.0.25.tar.gz            - ソース配布物
```

---

## ✨ 互換性

### パッケージ名

元のtorchcrepeと完全互換:
```python
import torchcrepe  # ← 同じimport名
```

### API互換性

全ての既存APIが動作:
```python
# 既存のコードはそのまま動作
pitch = torchcrepe.predict(audio, sr)  # デフォルトはfull

# 新しいモデルサイズも使用可能
pitch = torchcrepe.predict(audio, sr, model='medium')
```

---

## 🎯 次のステップ（オプション）

将来的な改善案:

1. **テストの追加**
   - tests/test_core.py に small/medium/large のテストケースを追加
   - 精度検証テスト（元のCREPEとの比較）

2. **ドキュメントの充実**
   - 各モデルサイズの推奨用途を記載
   - 速度vs精度のベンチマーク結果を追加

3. **パフォーマンス最適化**
   - TorchScriptによる高速化
   - ONNX形式への変換

4. **CI/CDの設定**
   - GitHub Actionsで自動テスト
   - 自動リリースパイプライン

---

## 📝 変更履歴

### Version 0.0.25 (2025-10-19)

**新機能:**
- ✨ small モデルサイズを追加 (6.23 MB)
- ✨ medium モデルサイズを追加 (22.44 MB)
- ✨ large モデルサイズを追加 (48.65 MB)

**変更:**
- 📝 README.md を更新してモデルサイズの説明を追加
- 📝 --model のヘルプテキストを更新

**技術:**
- 🔧 元のCREPE Keras重みをPyTorch形式に変換
- 🔧 torchcrepe.Crepe モデルを拡張

---

## 📞 サポート

### 問題が発生した場合

1. インポートエラー:
   ```bash
   pip install --force-reinstall dist/torchcrepe-0.0.25-py3-none-any.whl
   ```

2. モデルが見つからない:
   ```python
   import torchcrepe, os
   print(os.path.dirname(torchcrepe.__file__))  # assetsディレクトリを確認
   ```

3. 動作確認:
   ```bash
   python temp/test_installation.py
   ```

---

## 📄 ライセンス

MIT License (元のtorchcrepeと同じ)

---

**ビルド日時:** 2025-10-19
**Python バージョン:** 3.10
**PyTorch バージョン:** 2.5.1
