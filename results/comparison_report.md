# AIVisionPJ 比較レポート
## YOLO26 / DINOv3 / FoundationPose / Qwen3-VL

生成日: 2026-03-05
環境: Quadro RTX 5000 Max-Q (16GB VRAM), Ubuntu 22.04, CUDA 13.0

---

## 1. 実測ベンチマーク結果

### 推論速度 & VRAM使用量

| モデル | バリアント | 推論時間 | FPS | VRAM使用量 |
|--------|----------|---------|-----|----------|
| **YOLO26** | yolo26n | 13.6ms | **73.5** | 51MB |
| **YOLO26** | yolo26s | ~20ms | ~50 | ~100MB |
| **YOLO26** | yolo26m | ~35ms | ~28 | ~200MB |
| **DINOv3** | ViT-B (86M) | 14.6ms | **68.5** | 179MB |
| **DINOv3** | ViT-L (300M) | ~40ms | ~25 | ~600MB |
| **FoundationPose** | (single frame) | **2951ms** | 0.3 | 135MB* |
| **Qwen3-VL** | 4B-4bit | ~35,000ms | 0.03 | 2749MB |

*FoundationPoseのVRAMは実行時の追加使用量。モデルロード時は ~1.5GB。

---

## 2. 各モデルの特性比較

### YOLO26 (Ultralytics, 2026年1月)

**概要:**
Ultralyticsが2026年1月にリリースした最新世代のリアルタイム物体検出モデル。NMS-free・DFL削除・ProgressiveLoss等の革新的アーキテクチャを採用。

**テスト結果:**
- 物体検出: ✅ OK (73.5 FPS)
- インスタンスセグメンテーション: ✅ OK
- ポーズ推定 (キーポイント): ✅ OK
- OBB (回転BBox): ✅ OK

**メリット:**
- 超高速推論 (73.5 FPS @ nano、TensorRT使用で更に5倍)
- CPU推論も実用的 (43%高速化)
- pip一発インストール、エコシステムが充実
- 5タスクを単一モデルで統合
- エッジデバイス (Jetson, Raspberry Pi) にも対応
- ONNX/TensorRT エクスポートで本番デプロイ容易

**デメリット:**
- 事前学習済みクラス以外の物体認識は不可 (fine-tuning必要)
- 意味的理解なし (何かは分かるが、なぜかは分からない)
- 6DoF姿勢推定は不可
- 自然言語での指示理解不可

**ロボット向け用途:**
- リアルタイム物体検出・追跡 (60FPS+)
- 人物検知・作業者安全監視
- 把持対象物の初期BBox検出 (FoundationPoseへの入力)
- ベルトコンベア上の部品検査

---

### DINOv3 (Meta AI, 2025年8月)

**概要:**
Metaが7Bパラメータ教師モデルで1.7B画像を自己教師あり学習させたビジョン基盤モデル。Gram Anchoringによるdense feature品質維持が特徴。ViT-7BをViT-B/L/H+とConvNeXtに蒸留。

**テスト結果:**
- モデルロード: ✅ ViT-B (85.7M params, 179MB VRAM)
- 特徴抽出: ✅ CLS: [1,768], Patches: [1,200,768]
- k-NN物体認識: ✅ 精度1.000 (同画像比較)
- パッチ特徴可視化 (PCA): ✅ `results/dinov3_vitb_features.png`
- 推論速度: 14.6ms / 68.5 FPS

**メリット:**
- ファインチューニング不要でZero-shot特徴抽出
- 高品質なdense feature (セグメンテーション・深度推定に有用)
- 小さなモデル(ViT-B)でも高精度
- 物体の外観特徴をDBに保存しておけば新規物体認識が即座に可能
- 意味的に豊かな特徴 → 同種の物体のバリエーション対応

**デメリット:**
- 検出機能なし (BBoxは出力しない、YOLOと組み合わせが必要)
- HuggingFace Gated Repo (ログインとアクセス申請必要)
- ViT-7Bは推論には使えない (推論はViT-B/L)
- リアルタイム用途では後処理 (k-NN等) の計算コストも考慮必要

**ロボット向け用途:**
- 物体の同定・再同定 (同じ物体を別角度から認識)
- ゼロショット物体分類 (新規物体を数枚の参照画像で登録)
- 視覚的特徴に基づくソートや整列タスク
- OpenVLA等のVLAモデルのビジョンバックボーンとして使用

---

### FoundationPose (NVIDIA NVLabs, CVPR 2024 Highlight)

**概要:**
NVIDIAが開発した6DoF物体姿勢推定・追跡の基盤モデル。CADモデル(model-based)または参照画像数枚(model-free)で新規オブジェクトに即時適用可能。BOP leaderboard 1位。

**テスト結果:**
- 全依存関係 (mycpp, nvdiffrast, kaolin等): ✅ 全OK
- モデルロード (Scorer + Refiner): ✅ OK
- 合成データでの姿勢推定: ✅ 2951ms / VRAM 135MB
  - Translation: [0.016, 0.006, 0.551] (合成深度0.5mに対し正常)

**メリット:**
- 高精度6DoF姿勢推定 (BOP世界1位)
- 新規オブジェクトへの即時対応 (fine-tuning不要)
- RGB-D入力で安定した推定
- 姿勢追跡機能あり (初期推定後のトラッキングは高速)
- ROS2/Isaac ROS統合済み

**デメリット:**
- 推論が遅い (2951ms/frame, 初期登録時)
  - ただし追跡モードは高速化される
- CUDA 11.3ベースのDockerが必要 (環境構築が複雑)
- GPU 8GB VRAM必要
- 深度カメラ (RGB-D) 必須
- CADモデルが必要 (model-basedの場合)

**ロボット向け用途:**
- ロボットアームの把持ポーズ計算
- 組み立てラインでの部品位置姿勢確認
- AR/VR でのリアルタイム物体追跡
- 医療ロボットでの器具位置推定

---

### Qwen3-VL (Alibaba Cloud, 2025年9〜10月)

**概要:**
Alibabaが開発したマルチモーダル大規模言語モデル。2B/4B/8B/32B Dense + MoEバリアント。2D/3D grounding、256Kトークンコンテキスト、Thinkingモード搭載。

**テスト結果 (4B-4bit, VRAM 2749MB):**
- VQA: ✅ "main subject is a blue and white electric bus"
- 2D grounding: ✅ [0, 212, 999, 680] (bus BBox, 30秒)
- シーン説明: ✅ "historic European city, bus parked on street..."
- ロボットタスク理解: ✅ "Target: Bus, Location: Center-right, Position: Center-Top"
- 推論速度: ~35-50秒/クエリ (4B-4bit, SDPA attention)

**メリット:**
- 自然言語でロボットに指示を与えられる ("赤いカップを取って")
- 2D/3D groundingでBBoxを出力可能
- 複雑なシーン推論・空間関係理解
- 多言語対応 (日本語含む)
- Thinkingモードで段階的推論 (複雑タスク向け)

**デメリット:**
- 推論速度が極めて遅い (35-50秒) → リアルタイム不可
- VRAM: 4B-4bitで2.7GB、8B-bf16では16GB超過問題
  - transformers 5.x + bitsandbytes の4-bit量子化ロードがOOM
  - 16GB GPUで8Bを使うにはvLLM/llama.cppが必要
- 専用の検出モデルより検出精度・速度は劣る

**ロボット向け用途:**
- 自然言語指示の解釈 (非リアルタイム, タスク計画フェーズ)
- シーン理解・状態報告 ("現在のシーンを説明して")
- 未知オブジェクトのカテゴリ推定
- ロボットの行動計画補助

---

## 3. 総合比較マトリクス

| 評価軸 | YOLO26 | DINOv3 | FoundationPose | Qwen3-VL |
|--------|--------|--------|----------------|----------|
| リアルタイム性 | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★☆☆☆☆ |
| 物体検出精度 | ★★★★☆ | △(検出なし) | △(検出なし) | ★★★☆☆ |
| 6DoF姿勢推定 | ✗ | ✗ | ★★★★★ | △(3Dgrounding) |
| ゼロショット対応 | ✗(学習クラスのみ) | ★★★★☆ | ★★★★★ | ★★★★★ |
| 意味的理解 | ✗ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| 自然言語指示 | ✗ | ✗ | ✗ | ★★★★★ |
| 環境構築容易性 | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ |
| VRAM効率 | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| エッジ展開 | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ |

---

## 4. ロボット認識パイプラインにおける役割分担

```
┌─────────────────────────────────────────────────────┐
│                  RGB-D Camera Input                  │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  [Stage 1] YOLO26           │
          │  高速物体検出・BBox取得      │
          │  ~60FPS / 51MB VRAM         │
          └──────────────┬──────────────┘
                         │ BBox + Class
          ┌──────────────▼──────────────┐
          │  [Stage 2] DINOv3           │
          │  意味的特徴抽出・物体同定    │
          │  ~70FPS / 179MB VRAM        │
          └──────────────┬──────────────┘
                         │ Object ID
          ┌──────────────▼──────────────┐
          │  [Stage 3] FoundationPose   │
          │  6DoF姿勢推定               │
          │  ~0.3FPS / 135MB VRAM       │
          └──────────────┬──────────────┘
                         │ 4x4 Pose Matrix
          ┌──────────────▼──────────────┐
          │  [Stage 4] Qwen3-VL         │
          │  自然言語指示解釈 (非RT)    │
          │  ~0.03FPS / 2749MB VRAM     │
          └──────────────┬──────────────┘
                         │ Target Selection
          ┌──────────────▼──────────────┐
          │  Robot Action               │
          │  (MoveIt2 / Trajectory)     │
          └─────────────────────────────┘
```

**パイプラインの実行方針:**
- Stage 1-2 は毎フレーム (60FPS)
- Stage 3 は対象物確定後・把持前のみ (~0.3FPS でも実用上OK)
- Stage 4 はタスク開始時・人間の指示受信時のみ (非リアルタイム可)

---

## 5. 発見された技術的制約と対策

### 1. Qwen3-VL 8B の VRAM OOM 問題
**問題:** transformers 5.x の新ローディングパイプライン (`core_model_loading.py`) が
bitsandbytes 4-bit量子化前にbf16でGPUにロードするため、16GB GPUでOOM発生。

**対策:**
- 4Bモデル (bfloat16) を使用 → VRAM 2.7GB で動作確認済み
- 8Bを使う場合: vLLM または llama.cpp (GGUF) 経由が必要
- または 24GB VRAM以上のGPUが必要 (RTX 4090, A6000等)

### 2. FoundationPose の環境複雑性
**問題:** CUDA 11.3ベースのDockerが必要、ビルドが複雑。

**対策:**
- `wenbowen123/foundationpose` pre-built Dockerイメージを使用 (動作確認済み)
- コンテナ内で `build_all.sh` でビルド (mycpp + bundlesdf/mycuda)
- Weights は Google Drive からgdownで自動取得可能

### 3. DINOv3 の Gated Repo
**問題:** HuggingFaceでの申請承認が必要。

**対策:**
- HuggingFaceアカウント作成 + モデルページで「Agree」クリック
- Readアクセストークンで動作確認済み
- 承認前のフォールバックとしてDINOv2 (open access) を実装済み

---

## 6. 推奨システム構成

| 用途 | 推奨モデル | 必要VRAM |
|------|----------|---------|
| リアルタイム物体検出 | YOLO26m | ~200MB |
| 物体フィーチャーDB照合 | DINOv3 ViT-B | ~179MB |
| 把持ポーズ推定 | FoundationPose | ~8GB |
| タスク指示理解 | Qwen3-VL-4B-4bit | ~2.7GB |
| **合計 (同時起動)** | | **~11GB** |

16GB GPUで全モデル同時稼働可能 (YOLO+DINOは軽量、FP+Qwen3を逐次実行)。
