================================================================================
torchcrepe拡張開発者ガイド
small, medium, large モデルサイズの追加実装について
================================================================================

【結論】
コードの変更だけでは実装不可能です。
事前学習済み重みファイル (.pth形式) の入手と変換が必要です。

================================================================================
【現状分析】
================================================================================

1. 現在のtorchcrepeには以下の2モデルのみ実装:
   - tiny (capacity_multiplier: 4)
   - full (capacity_multiplier: 32)

2. 必要なモデルサイズ:
   - small  (capacity_multiplier: 8)  - 2/8相当
   - medium (capacity_multiplier: 16) - 4/8相当
   - large  (capacity_multiplier: 24) - 6/8相当

3. 現在のアセット状況:
   torchcrepe/assets/
   ├── full.pth  (85MB) - 存在
   └── tiny.pth  (1.9MB) - 存在

   不足しているファイル:
   ├── small.pth  - 不足
   ├── medium.pth - 不足
   └── large.pth  - 不足

================================================================================
【モデルアーキテクチャ定義】（元のCREPEより）
================================================================================

capacity_multiplier = {
    'tiny': 4,
    'small': 8,
    'medium': 16,
    'large': 24,
    'full': 32
}

各レイヤーのチャネル数 = [32, 4, 4, 4, 8, 16] × capacity_multiplier

計算結果:
- tiny   (4):  [128,  16,  16,  16,  32,  64]  ✓実装済み
- small  (8):  [256,  32,  32,  32,  64, 128]  ×未実装
- medium (16): [512,  64,  64,  64, 128, 256]  ×未実装
- large  (24): [768,  96,  96,  96, 192, 384]  ×未実装
- full   (32): [1024, 128, 128, 128, 256, 512] ✓実装済み

in_features (最終分類層前の特徴量次元):
- tiny:   256
- small:  512
- medium: 1024
- large:  1536
- full:   2048

================================================================================
【実装方法 - 3つのアプローチ】
================================================================================

----------------------------------------
方法1: 元のCREPE (TensorFlow/Keras) から変換【推奨】
----------------------------------------

手順:
1. 元のCREPEから.h5重みファイルをダウンロード
   URL: https://github.com/marl/crepe/raw/models/model-{capacity}.h5.bz2
   - model-small.h5.bz2
   - model-medium.h5.bz2
   - model-large.h5.bz2

2. .bz2ファイルを展開
   ```bash
   bunzip2 model-small.h5.bz2
   bunzip2 model-medium.h5.bz2
   bunzip2 model-large.h5.bz2
   ```

3. Keras (.h5) → PyTorch (.pth) 変換

   必要なツール:
   - MMdnn (元のtorchcrepe作者が使用したツール)
     https://github.com/microsoft/MMdnn

   または

   - 手動変換スクリプト作成
     参考: https://github.com/AgCl-LHY/Weights_Keras_2_Pytorch

   変換時の注意点:
   - Conv2D: weight shape変換 (H,W,C_in,C_out) → (C_out,C_in,H,W)
   - BatchNorm: running_mean, running_var, weight, bias の4パラメータ
   - Dense: weight transpose (in,out) → (out,in)
   - レイヤー名を元のCREPEと一致させる必要あり

4. 変換した.pthファイルをtorchcrepe/assets/に配置
   ```
   torchcrepe/assets/
   ├── small.pth
   ├── medium.pth
   └── large.pth
   ```

メリット:
- 公式の重みを使用できる（最も信頼性が高い）
- 元のCREPE論文の結果を再現可能

デメリット:
- 変換作業が複雑
- TensorFlow/Kerasの環境構築が必要

----------------------------------------
方法2: ONNX形式から変換
----------------------------------------

手順:
1. onnxcrepeのリリースから.onnxファイルをダウンロード
   URL: https://github.com/yqzhishen/onnxcrepe/releases
   - small.onnx  (約8MB)
   - medium.onnx (約17MB)
   - large.onnx  (約51MB)

2. ONNX → PyTorch変換
   ```python
   import torch
   import onnx
   from onnx2pytorch import ConvertModel

   # ONNXモデルをロード
   onnx_model = onnx.load('small.onnx')

   # PyTorchモデルに変換
   pytorch_model = ConvertModel(onnx_model)

   # state_dictを保存
   torch.save(pytorch_model.state_dict(), 'small.pth')
   ```

   必要なライブラリ:
   - onnx
   - onnx2pytorch (pip install onnx2pytorch)

3. 変換した.pthファイルを配置

メリット:
- .onnxファイルは既に用意されている
- 変換が比較的簡単

デメリット:
- ONNX → PyTorch変換でモデル構造が一致しない可能性
- 追加の検証が必要

----------------------------------------
方法3: 他のPyTorch実装から流用
----------------------------------------

以下のリポジトリを確認:
- https://github.com/sweetcocoa/crepe-pytorch
  → small/medium/largeのコード実装はあるが、.pth重みは未確認

手順:
1. 他のPyTorch実装のリポジトリを探索
2. .pthファイルが公開されているか確認
3. ライセンスを確認して使用可能か判断
4. 互換性を検証してコピー

メリット:
- すぐに使える可能性

デメリット:
- 公開されている保証がない（現時点で発見できず）
- ライセンス問題の可能性

================================================================================
【実装手順（重みファイル入手後）】
================================================================================

1. torchcrepe/model.py を編集

   Crepe.__init__() メソッドを以下のように修正:

   ```python
   def __init__(self, model='full'):
       super().__init__()

       # Model-specific layer parameters
       if model == 'full':
           in_channels = [1, 1024, 128, 128, 128, 256]
           out_channels = [1024, 128, 128, 128, 256, 512]
           self.in_features = 2048
       elif model == 'large':
           in_channels = [1, 768, 96, 96, 96, 192]
           out_channels = [768, 96, 96, 96, 192, 384]
           self.in_features = 1536
       elif model == 'medium':
           in_channels = [1, 512, 64, 64, 64, 128]
           out_channels = [512, 64, 64, 64, 128, 256]
           self.in_features = 1024
       elif model == 'small':
           in_channels = [1, 256, 32, 32, 32, 64]
           out_channels = [256, 32, 32, 32, 64, 128]
           self.in_features = 512
       elif model == 'tiny':
           in_channels = [1, 128, 16, 16, 16, 32]
           out_channels = [128, 16, 16, 16, 32, 64]
           self.in_features = 256
       else:
           raise ValueError(f'Model {model} is not supported')

       # ... (残りのコードは同じ)
   ```

2. torchcrepe/__main__.py を編集

   argparserのhelpテキストを更新:

   ```python
   parser.add_argument(
       '--model',
       default='full',
       help='The model capacity. One of "tiny", "small", "medium", "large", or "full"')
   ```

3. setup.py を確認

   package_dataに新しいアセットが含まれることを確認:

   ```python
   package_data={'torchcrepe': ['assets/*']}
   ```

4. README.md を更新

   モデルサイズの説明を追加

5. テストの追加/更新

   tests/test_core.py に新しいモデルサイズのテストを追加:

   ```python
   def test_embed_small(audio):
       """Tests that the embedding is the expected size"""
       embedding = torchcrepe.embed(audio, torchcrepe.SAMPLE_RATE, 160, 'small')
       assert embedding.size() == (1, 1001, 32, 16)  # 16 = 512/32

   def test_embed_medium(audio):
       embedding = torchcrepe.embed(audio, torchcrepe.SAMPLE_RATE, 160, 'medium')
       assert embedding.size() == (1, 1001, 32, 32)  # 32 = 1024/32

   def test_embed_large(audio):
       embedding = torchcrepe.embed(audio, torchcrepe.SAMPLE_RATE, 160, 'large')
       assert embedding.size() == (1, 1001, 32, 48)  # 48 = 1536/32
   ```

6. テスト実行

   ```bash
   pytest tests/test_core.py -v
   ```

7. パッケージング

   ```bash
   # バージョン番号を更新 (setup.py)
   version='0.0.25'

   # ビルド
   pip install build
   python -m build

   # whlファイルが生成される
   # dist/torchcrepe-0.0.25-py3-none-any.whl
   ```

8. ローカルインストールして検証

   ```bash
   pip install dist/torchcrepe-0.0.25-py3-none-any.whl

   # 動作確認
   python -c "import torchcrepe; print(torchcrepe.Crepe('small'))"
   python -m torchcrepe --model small --audio_files test.wav --output_files pitch.pt
   ```

================================================================================
【推奨実装フロー】
================================================================================

フェーズ1: 重みファイルの入手 (最も重要)
  □ 元のCREPEリポジトリから.h5ファイルをダウンロード
    - wget https://github.com/marl/crepe/raw/models/model-small.h5.bz2
    - wget https://github.com/marl/crepe/raw/models/model-medium.h5.bz2
    - wget https://github.com/marl/crepe/raw/models/model-large.h5.bz2
  □ 解凍
    - bunzip2 model-*.h5.bz2

フェーズ2: 重みファイルの変換
  オプションA: MMdnnを使用
    □ MMdnnのインストール
    □ Keras→PyTorch変換スクリプト作成
    □ 変換実行

  オプションB: 手動変換スクリプト
    □ TensorFlow/Keras環境構築
    □ 変換スクリプト作成 (Weights_Keras_2_Pytorchを参考)
    □ レイヤー名マッピング確認
    □ 変換実行

フェーズ3: コード実装
  □ model.pyにsmall/medium/large定義追加
  □ __main__.pyのヘルプ更新
  □ README.md更新

フェーズ4: テストと検証
  □ テストケース追加
  □ pytest実行
  □ 手動テスト（実際の音声ファイルで確認）
  □ 元のCREPEと精度比較（オプション）

フェーズ5: パッケージング
  □ setup.pyのバージョン更新
  □ python -m build 実行
  □ whlファイル生成確認
  □ 他プロジェクトでインストールテスト

================================================================================
【必要なツールとライブラリ】
================================================================================

変換用:
- TensorFlow 2.x / Keras
- h5py
- MMdnn または onnx2pytorch
- numpy

開発用:
- PyTorch
- pytest (テスト用)
- build (パッケージング用)

================================================================================
【参考リンク】
================================================================================

元のCREPEリポジトリ:
https://github.com/marl/crepe

重みファイルダウンロード:
https://github.com/marl/crepe/tree/models

onnxcrepe (変換済みONNXモデルあり):
https://github.com/yqzhishen/onnxcrepe

Keras→PyTorch変換スクリプト:
https://github.com/AgCl-LHY/Weights_Keras_2_Pytorch

MMdnn (モデル変換ツール):
https://github.com/microsoft/MMdnn

================================================================================
【見積もり作業時間】
================================================================================

重みファイル変換（初めての場合）: 4-8時間
  - 環境構築: 1-2時間
  - 変換スクリプト作成/デバッグ: 2-4時間
  - 検証: 1-2時間

コード実装: 1-2時間

テスト作成/実行: 1-2時間

ドキュメント更新: 0.5-1時間

合計: 6.5-13時間

================================================================================
【トラブルシューティング】
================================================================================

問題: 変換した重みをロードできない
→ レイヤー名が一致しているか確認
→ shape変換が正しいか確認 (Conv2Dは(H,W,C_in,C_out)→(C_out,C_in,H,W))

問題: テストで精度が元のCREPEと異なる
→ BatchNormのrunning_mean/varが正しく変換されているか確認
→ Dropout層は推論時に無効化されているか確認

問題: whlインストール後にアセットが見つからない
→ setup.pyのpackage_dataが正しいか確認
→ MANIFEST.inが必要な場合あり

================================================================================
最終更新: 2025-10-19
================================================================================
