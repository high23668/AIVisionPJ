# AIVisionPJ

ロボット認識パイプライン研究プロジェクト。  
**YOLO26 / DINOv3 / SAM3 / FoundationPose / Qwen3-VL / Any6D / ASGrasp / FoundationStereo / Boxer / Depth Pro** を比較・統合し、透明物体 (ガラス板) の6DoF姿勢推定パイプラインを構築する。

---

## 目次

1. [環境](#環境)
2. [モデル一覧・ベンチマーク](#モデル一覧ベンチマーク)
3. [各モデルの使い方](#各モデルの使い方)
4. [確立パイプライン (透明物体認識)](#確立パイプライン-透明物体認識)
5. [ガラス板実験 結果サマリ (Phase 1-7)](#ガラス板実験-結果サマリ-phase-1-7)
6. [第2段実験 計画](#第2段実験-計画)
7. [スクリプト一覧](#スクリプト一覧)
8. [既知の罠・トラブルシューティング](#既知の罠トラブルシューティング)
9. [進捗チェックリスト](#進捗チェックリスト)

---

## 環境

### ハードウェア / OS
| 項目 | 内容 |
|------|------|
| GPU | Quadro RTX 5000 Max-Q (16GB VRAM, SM 7.5 Turing) |
| OS | Ubuntu 22.04 |
| CUDA | 13.0 |
| カメラ | Intel RealSense D435i |

### Python 仮想環境

| 環境 | Python | 用途 | アクティベート |
|------|--------|------|--------------|
| `.venv` | 3.10 | YOLO26 / DINOv3 / FoundationPose / Qwen3-VL / RealSense | `source .venv/bin/activate` |
| `.venv_sam3` | 3.12 | SAM3 専用 | `source .venv_sam3/bin/activate` |
| `.venv_3d` | 3.10 | SF3D (Stable Fast 3D) / Any6D メッシュ生成 | `source .venv_3d/bin/activate` |
| `.venv_asgrasp` | - | ASGrasp | `source .venv_asgrasp/bin/activate` |
| `.venv_stereo` | - | Foundation Stereo | `source .venv_stereo/bin/activate` |

パッケージ管理は **uv** を使用 (conda 未インストール)。

### Docker

| コンテナ | イメージ | 用途 |
|---------|---------|------|
| `foundationpose_build` | `wenbowen123/foundationpose` (31.7GB) | FoundationPose / Any6D 推論 |
| `any6d:latest` | カスタムビルド | Any6D (SF3D → FoundationPose) |

**Docker 起動共通手順:**
```bash
docker start foundationpose_build
docker exec -it foundationpose_build bash -c '
  source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
  export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
'
```

---

## モデル一覧・ベンチマーク

### 動作確認済みモデル

| モデル | バリアント | 推論時間 | FPS | VRAM | 環境 | 確認日 |
|--------|----------|---------|-----|------|------|-------|
| **YOLO26** | yolo26n | 13.6ms | 73.5 | 51MB | `.venv` | 2026-03-05 |
| **DINOv3** | ViT-B | 14.6ms | 68.5 | 179MB | `.venv` | 2026-03-05 |
| **SAM3** | - | ~1700ms | 0.6 | - | `.venv_sam3` | 2026-03-06 |
| **FoundationPose** | model-based | 2951ms | 0.3 | 135MB* | Docker | 2026-03-05 |
| **Qwen3-VL** | 4B-4bit | ~35,000ms | 0.03 | 2749MB | `.venv` | 2026-03-05 |
| **SF3D** | - | ~1,000ms | 1.0 | 6168MB | `.venv_3d` | 2026-03-25 |
| **Any6D + SF3D** | - | - | - | - | Docker | 2026-03-28 |
| **ASGrasp** | - | - | - | - | `.venv_asgrasp` | 2026-03-29 |
| **Depth Pro** | Apple | - | - | - | `.venv` | 2026-05-02 |

*FoundationPose の VRAM はモデルロード ~1.5GB + 推論時追加 135MB。

### 比較マトリクス

| 評価軸 | YOLO26 | DINOv3 | SAM3 | FoundationPose | Qwen3-VL | Any6D |
|--------|:------:|:------:|:----:|:--------------:|:--------:|:-----:|
| リアルタイム性 | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ |
| 物体検出精度 | ★★★★☆ | △ | △ | △ | ★★★☆☆ | △ |
| 6DoF 姿勢推定 | ✗ | ✗ | ✗ | ★★★★★ | △ | ★★★★☆ |
| ゼロショット対応 | ✗ | ★★★★☆ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| CADメッシュ不要 | ✗ | - | - | ✗ | - | ✅ |
| 自然言語指示 | ✗ | ✗ | △ | ✗ | ★★★★★ | ✗ |
| VRAM 効率 | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |

### ロボットパイプラインにおける役割

```
RGB-D Camera Input
       │
  [Stage 1] YOLO26       : 高速物体検出・BBox (60+ FPS)
       │ BBox + Class
  [Stage 2] DINOv3        : 意味的特徴抽出・物体同定 (~70 FPS)
       │ Object ID
  [Stage 3] SAM3          : ピクセル精度マスク生成
       │ Mask
  [Stage 4] FoundationPose: 6DoF 姿勢推定 (把持前のみ)
       │ 4×4 Pose Matrix
  [Stage 5] Qwen3-VL      : 自然言語指示解釈 (タスク開始時のみ)
       │ Target Selection
  Robot Action (MoveIt2 / Trajectory)
```

---

## 各モデルの使い方

### YOLO26 - 高速物体検出

```bash
source .venv/bin/activate

# RealSense カメラでリアルタイム検出
yolo detect predict model=yolo26n.pt source=4 show=True

# 信頼度しきい値を上げる (デフォルト 0.25)
yolo detect predict model=yolo26n.pt source=4 show=True conf=0.5

# 特定クラスのみ (person=0, bottle=39, cup=41)
yolo detect predict model=yolo26n.pt source=4 show=True classes=0,39,41

# デモスクリプト
python models/yolo26/yolo26_demo.py
```

**カメラ source 番号の確認:**
```bash
python -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w, h = cap.get(3), cap.get(4)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'source={i}: {int(w)}x{int(h)} @ {fps:.0f}fps')
        cap.release()
" 2>/dev/null
```
`640x480 @ 30fps` でカラー映像が出る番号を使う (前回確認: `source=4`)。

**COCO 主要クラス番号:**

| 番号 | クラス | 番号 | クラス |
|------|--------|------|--------|
| 0 | person | 39 | bottle |
| 14 | bird | 41 | cup |
| 15 | cat | 56 | chair |
| 16 | dog | 63 | laptop |
| 26 | handbag | 67 | cell phone |

---

### DINOv3 - ゼロショット物体認識

アノテーション・学習なしで参照画像を登録するだけで新規物体を認識できる。

```bash
source .venv/bin/activate

# Step 1: 物体を登録
python models/dinov3/dinov3_demo.py --task register --label cup --source 4
python models/dinov3/dinov3_demo.py --task register --label pencil --source 4

# Step 2: ライブ認識
python models/dinov3/dinov3_demo.py --task recognize --source 4

# 登録リセット
rm -f results/gallery.npz results/ref_*.jpg
```

**HuggingFace 認証** (初回 or トークン切れ時):
```bash
python -c "from huggingface_hub import login; login(token='<your_token>')"
```
※ HF Gated Repo → モデルページで「Agree」が必要 (承認済み)。

---

### SAM3 - テキストプロンプト セグメンテーション

```bash
source .venv_sam3/bin/activate

# 1枚撮影してセグメント
DISPLAY=:0 python models/sam3/sam3_demo.py --task camera --source 4 --prompt "cup"

# ライブモード (プロンプト入力→撮影→表示を繰り返す)
DISPLAY=:0 python models/sam3/sam3_demo.py --task live --source 4

# 静止画ファイル
DISPLAY=:0 python models/sam3/sam3_demo.py --task image --image data/cup.png --prompt "bottle"

# ストリーミング (~0.6FPS) → ブラウザで http://localhost:8080
DISPLAY=:0 python models/sam3/sam3_demo.py --task stream --source 4 --prompt "cup"
```

**注意:** VSCode ターミナルでは `DISPLAY=:0` が必須。CUDA 推論後の `cv2.imshow` はセグフォルトするため結果は `xdg-open` で表示。

---

### FoundationPose - 6DoF 姿勢推定

**重要:** model-free は事実上使用不可 (参照画像の既知カメラポーズ + BundleSDF が必要)。**model-based (CAD メッシュあり) を使う。**

```bash
# Step 1: RealSense でクエリ画像撮影
source .venv/bin/activate
python scripts/capture_realsense.py
# → data/realsense/rgb.png, depth.png, depth.npy, intrinsics.json

# Step 2: SAM3 でマスク生成
source .venv_sam3/bin/activate
DISPLAY=:0 python scripts/generate_sam3_mask.py --image data/realsense/rgb.png --prompt "cup"

# Step 3: FP 用シーンデータ整理
source .venv/bin/activate
python scripts/setup_fp_scene.py

# Step 4: Docker 内で FoundationPose 実行
docker start foundationpose_build
docker exec -it foundationpose_build bash -c '
  source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
  export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH &&
  FP_SKIP_IMSHOW=1 python /run_demo.py \
    --mesh_file data/cad_models/object.obj \
    --test_scene_dir data/fp_scene \
    --debug_dir results/fp_debug
'
```

**CAD メッシュの入手方法 (model-based 必須):**
- **PolyCam** (iOS/Android) でスキャン → OBJ エクスポート
- **MeshRoom** (オープンソース) でフォトグラメトリ
- 生成した `.obj` を `data/cad_models/` に配置

---

### Qwen3-VL - マルチモーダル LLM

```bash
source .venv/bin/activate

# VQA デモ
python models/qwen3vl/qwen3vl_demo.py --task vqa --image data/cup.png \
  --question "What is the main object?"

# 2D Grounding (BBox 出力)
python models/qwen3vl/qwen3vl_demo.py --task ground --image data/cup.png \
  --question "Locate the cup"
```

**注意:** 8B は 16GB VRAM で OOM (transformers 5.x + bitsandbytes の問題) → **4B-4bit を使用**。速度は ~35秒/クエリ。

---

### Any6D + SF3D - CAD 不要 6DoF 推定

**単一 RGBD 画像 1 枚**でモデルフリー 6DoF 推定。CAD メッシュ不要。  
構成: SAM2 → SF3D (InstantMesh の代替、VRAM ~6GB) → FoundationPose。

```bash
# Step 1: RealSense で撮影
source .venv/bin/activate
python scripts/capture_realsense.py

# Step 2: SAM3 でマスク生成
source .venv_sam3/bin/activate
DISPLAY=:0 python scripts/generate_sam3_mask.py --image data/realsense/rgb.png --prompt "cup"

# Step 3: SF3D でメッシュ生成 (ホスト側)
source .venv_3d/bin/activate
python scripts/any6d_generate_mesh.py --image data/realsense/rgb.png

# Step 4: Docker 内で Any6D 実行
bash scripts/any6d_pipeline.sh
```

**align_mesh scale=0.1:** SF3D の生出力が ~90cm → 実物サイズに調整。  
結果: `results/any6d_realsense/pose_overlay.png`, `pred_pose.txt`

---

### Depth Pro (Apple) - 単眼深度推定

透明ガラス板の深度復元で **現時点ベスト** (Phase 7 実証済み)。

```bash
source .venv/bin/activate

python scripts/phase7_depthpro.py \
  --image data/glassboard_simple/rgb.png \
  --output results/depth_pro_output.npy
```

チェックポイント: `models/depth_pro/checkpoints/depth_pro.pt`

---

### Foundation Stereo - ステレオ深度推定

RealSense の IR ステレオペアから高精度 depth を推定。

```bash
source .venv_stereo/bin/activate
python models/foundation_stereo/FoundationStereo/demo.py \
  --left data/glassboard_simple/left_ir.png \
  --right data/glassboard_simple/right_ir.png
```

**限界:** 透明板の "面" は不可視、エッジのみ復元 (Phase 3 確認済み)。

---

### ASGrasp - 透明物体把持

DREDS データセット学習済みの透明物体把持位置推定モデル。

```bash
source .venv_asgrasp/bin/activate
python models/asgrasp/ASGrasp/run_demo.py \
  --rgb data/glassboard_simple/rgb.png \
  --depth data/glassboard_simple/depth.npy
```

**限界:** ガラス板 (厚み 2.5mm) は物理的に掴めないため grasp 対象外と判定される (Phase 5 確認済み)。

---

### Boxer - 3D OBB 推定

RGB + Depth から 3D 有向境界ボックスを推定。

```bash
source models/boxer/.venv/bin/activate
python models/boxer/run_boxer.py \
  --image data/glassboard_simple/rgb.png
```

---

## 確立パイプライン (透明物体認識)

Phase 7 で実証済み。ガラス板 148×148×2.5mm の 6DoF 推定に成功 (7/8)。

```
[1] 撮影
    python scripts/capture_realsense.py
    → data/realsense/{rgb.png, depth.npy, intrinsics.json}

[2] マスク生成 (SAM3 最大面積マスク)
    DISPLAY=:0 python scripts/generate_sam3_mask.py \
      --image data/realsense/rgb.png --prompt "glass" --mode largest_area
    → data/realsense/mask.png

[3] 単眼深度推定 (Depth Pro 推奨)
    python scripts/phase7_depthpro.py \
      --image data/realsense/rgb.png
    → depth_depthpro.npy

[4] Scale 校正 (RS 壁面基準)
    scale = median(RS_wall_depth) / median(model_wall_depth)
    depth_calibrated = depth_depthpro * scale

[5] FP 用 uint16 PNG 変換
    (depth_calibrated * 1000).astype(uint16) → depth_uint16.png

[6] FoundationPose 実行
    FP_SKIP_IMSHOW=1 python run_demo.py \
      --mesh_file data/glassboard_cad/glass_plate.obj \
      --test_scene_dir data/fp_scene
    → 4×4 Pose Matrix
```

**SAM3 のプロンプト注意:** `glass` プロンプトの最高スコアはペン瓶 (0.71) になる場合がある → **最大面積マスクを選ぶ**。

---

## ガラス板実験 結果サマリ (Phase 1-7)

**対象:** ガラス板 148×148×2.5mm / RealSense D435i / simple・complex 2シーン  
**データ:** `data/glassboard_{simple,complex}/` / CAD: `data/glassboard_cad/glass_plate.obj`  
**レポート:** `results/glassboard_experiment_report.md`

| Phase | 手法 | 主要結果 |
|-------|------|---------|
| **Phase 1** | YOLO26 / SAM3 / Qwen3-VL | YOLO26 完敗。SAM3 形状◎・材質識別×。Qwen3-VL がディストラクタ識別 MVP |
| **Phase 2** | RS depth × Boxer / Any6D | 厚み 17cm 過大。Any6D Simple は板にフィット |
| **Phase 3** | Foundation Stereo | 板の "面" は不可視、エッジのみ復元 |
| **Phase 4** | FS depth × Boxer / Any6D | Phase 2 と同等、根本限界変わらず |
| **Phase 5** | ASGrasp | 板を grasp 対象とせず (厚み 2.5mm + 学習バイアス) |
| **Phase 6** | DA2 / Marigold / FP×CAD | DA2 が板面を初めて密に復元。FP×CAD は 0/4 失敗 |
| **Phase 7** | Metric3D v2 / MoGe-2 / Depth Pro / UniDepth V2 | **FP×CAD: 7/8 成功** |

### Phase 7 詳細結果

| モデル | 板の面 depth | metric | 境界 | FP×CAD 成功 |
|--------|------------|--------|------|-----------|
| Metric3D v2 ViT-L | ✗ 壁に溶込み | ◯ | △ | 1/2 |
| MoGe-2 ViT-L + normal | ✅ 壁より前 | ◯ | ◯ | 2/2 |
| **Depth Pro (Apple)** | ✅ **最鮮明** | ◯ | ✅ | **2/2** |
| UniDepth V2 ViT-L | ◯ 弱め | ◯ + conf | △ | 2/2 |

**結論:** Depth Pro が現時点ベスト (境界最鮮明・Simple/Complex とも完全フィット)。  
Metric3D v2 は透明板を背景と同一視するため不適。

---

## 第2段実験 計画

**計画書:** `docs/exp2/plan_exp2.md`  
**目的:** Phase 1 の課題 (GT なし / マスク問題 / scale 校正不正確) を解消した上で全 15 モデルを定量比較。

### Phase 1 からの改善点

| 課題 | 改善策 |
|------|--------|
| Ground Truth がない | カラー紙をワークに貼って RS で板面 GT 取得 |
| Scale 校正が不正確 | GT Z で直接校正 (壁基準から脱却) |
| Complex マスク問題 | カラー紙状態で SAM3 → 透明状態に流用 |
| 定量評価なし | RMSE / MAE / Inlier rate / FP Z 誤差 を CSV 出力 |

### 評価対象 (全 15 モデル)

| カテゴリ | モデル |
|---------|--------|
| RGB 系 | YOLO26, DINOv3, SAM3, Qwen3-VL 4B |
| Depth 系 | DA2 Indoor Metric, Marigold v1.1, Metric3D v2, MoGe-2, **Depth Pro**, UniDepth V2 |
| Stereo / Grasp | Foundation Stereo, ASGrasp |
| 3D ポーズ推定 | Boxer, Any6D (SF3D), FoundationPose × parametric CAD |

### ワーク

| ワーク | サイズ | CAD |
|--------|--------|-----|
| ガラス板 | 148×148×2.5mm | `data/exp2/cad/glass_plate.obj` |
| 透明樹脂シート | 70×100×0.1mm | `data/exp2/cad/resin_sheet.obj` (新規作成) |

---

## スクリプト一覧

### scripts/

| スクリプト | 用途 | 環境 |
|-----------|------|------|
| `capture_realsense.py` | RealSense で 1 フレーム撮影 | `.venv` |
| `capture_realsense_dual.py` | RGB + IR ステレオ同時撮影 | `.venv` |
| `capture_ref_images.py` | 参照画像セット撮影 (インタラクティブ) | `.venv` |
| `generate_sam3_mask.py` | SAM3 でバイナリマスク生成 | `.venv_sam3` |
| `setup_fp_scene.py` | FoundationPose 用データ整理 | `.venv` |
| `fp_create_mesh_and_run.py` | FoundationPose 実行 | Docker |
| `any6d_generate_mesh.py` | SF3D でメッシュ生成 | `.venv_3d` |
| `any6d_pipeline.sh` | Any6D フルパイプライン | `.venv_3d` + Docker |
| `phase7_depthpro.py` | Depth Pro 深度推定 | `.venv` |
| `phase7_moge2.py` | MoGe-2 深度推定 | `.venv` |
| `phase7_unidepth.py` | UniDepth V2 深度推定 | `.venv` |
| `phase7_metric3d.py` | Metric3D v2 深度推定 | `.venv` |
| `phase7_run_foundationpose.sh` | FP バッチ実行 | Docker |
| `warp_fs_depth_to_rgb.py` | Foundation Stereo depth を RGB に投影 | `.venv_stereo` |

### データディレクトリ構造

```
data/
├── realsense/              # RealSense キャプチャ (最新)
│   ├── rgb.png             # クエリ画像
│   ├── depth.png           # 深度 (uint16 mm)
│   ├── depth.npy           # 深度 (float32 meters)
│   ├── mask.png            # SAM3 マスク (binary)
│   └── intrinsics.json     # カメラ内部パラメータ
├── glassboard_simple/      # ガラス板 simple シーン
├── glassboard_complex/     # ガラス板 complex シーン (ディストラクタあり)
├── glassboard_cad/
│   ├── glass_plate.obj     # ガラス板 CAD (148×148×2.5mm)
│   └── glass_plate.ply
├── exp2/                   # 第2段実験データ
│   ├── cad/
│   ├── glass_front/
│   ├── glass_oblique/
│   ├── resin_front/
│   └── resin_oblique/
├── fp_scene/               # FoundationPose 入力形式
│   ├── rgb/000000.png
│   ├── depth/000000.png
│   ├── masks/000000.png
│   └── cam_K.txt
└── cad_models/             # CAD メッシュ置き場 (任意物体)
```

---

## 既知の罠・トラブルシューティング

### FoundationPose

| 問題 | 対策 |
|------|------|
| `cv2.imshow` が Docker でクラッシュ (SIGABRT) | `FP_SKIP_IMSHOW=1` 環境変数を設定。`try/except` では防げない |
| `QT_QPA_PLATFORM=offscreen` を設定すると起動失敗 | Docker コンテナに offscreen プラグインがない。`FP_SKIP_IMSHOW=1` を使う |
| depth 入力形式エラー | uint16 PNG (mm 単位) が必須。float メートル値は `* 1000` して `uint16` 化 |

### SAM3

| 問題 | 対策 |
|------|------|
| セグフォルト (CUDA 推論後の `cv2.imshow`) | 結果の表示は `xdg-open` を使う |
| `GatedRepoError` / `awaiting a review` | Meta 側承認待ち。承認後 `.venv_sam3` で HF ログイン |
| VSCode ターミナルで実行するとウィンドウが出ない | `DISPLAY=:0` を付けて実行 |

### DINOv3

| 問題 | 対策 |
|------|------|
| `GatedRepoError` | HF ログインと承認が必要 (承認済み) |
| トークン切れ | `.venv` で `from huggingface_hub import login; login(token='...')` |

### SF3D

| 問題 | 対策 |
|------|------|
| 横長画像でクラッシュ | 入力は正方形前提。`crop_to_square()` でオブジェクト中心クロップ |
| GLB 読み込みで `Scene` が返る | `scene.dump()` + `concatenate` してから export |
| Flash Attention で bfloat16 エラー | SM 7.5 (Turing) は float16 のみ対応。`autocast(dtype=torch.float16)` を使う |

### MoGe-2

| 問題 | 対策 |
|------|------|
| depth が ~63m になる | `fov_x` 引数は **degrees 単位**。radians を渡すと誤推定 |

### Qwen3-VL

| 問題 | 対策 |
|------|------|
| 8B が 16GB で OOM | transformers 5.x の新ローディングが bfloat16 でフル GPU ロードするため。**4B-4bit を使用** |
| グラウンディング精度が低い | 4B の限界。8B には vLLM / llama.cpp (GGUF) が必要 |

### カメラ関連

| 問題 | 対策 |
|------|------|
| 画像が真っ暗 / 真緑 | auto-exposure が未安定。再実行する |
| `source=X: cannot open` | USB 接続を確認後 `rs-enumerate-devices` を実行 |
| RealSense depth が真っ黒に見える | uint16 の値が小さい (max ~1660) のは正常。値は正しく記録されている |

---

## 進捗チェックリスト

### 完了済み

- [x] YOLO26: 実装・テスト完了 (73.5 FPS / 51MB VRAM) ─ 2026-03-05
- [x] DINOv3: 実装・テスト完了 (68.5 FPS / 179MB VRAM) ─ 2026-03-05
- [x] FoundationPose: Docker 環境・動作確認完了 ─ 2026-03-05
- [x] Qwen3-VL 4B: 実装・テスト完了 (~35s / 2749MB VRAM) ─ 2026-03-05
- [x] SAM3: 実装・テスト完了 (~1700ms / Meta HF 承認済み) ─ 2026-03-06
- [x] FoundationPose + RealSense パイプライン構築完了 ─ 2026-03-13
- [x] Any6D + Image-to-3D 技術調査完了 ─ 2026-03-24
- [x] SF3D 単体テスト完了 (推論~1秒 / 6168MB VRAM) ─ 2026-03-25
- [x] Any6D + SF3D 統合・動作確認完了 ─ 2026-03-28
- [x] Any6D RealSense 実物テスト完了 (スヌーピーマグカップ) ─ 2026-03-29
- [x] ガラス板実験 Phase 1〜7 全完了 ─ 2026-05-02〜03
  - [x] Phase 7: Depth Pro / MoGe-2 / UniDepth V2 で FP×CAD 7/8 成功
  - [x] **確立パイプライン:** SAM3 最大面積マスク → Depth Pro → RS 壁面 scale 校正 → FP × parametric CAD
- [x] 第2段実験 計画策定 (`docs/exp2/plan_exp2.md`) ─ 2026-05-03

### 未完了

- [ ] **第2段実験 実施** (15 モデル × 4 シーン、定量評価)
  - [ ] 透明樹脂シート入手 (70×100mm PET)
  - [ ] 樹脂シート CAD 作成 (`data/exp2/cad/resin_sheet.obj`)
  - [ ] `scripts/exp2/` スクリプト整備
  - [ ] 撮影 (8 シーン × 4 ショット)
  - [ ] 全モデル実行 → `results/exp2/metrics/` に CSV 出力
- [ ] FoundationPose model-based: 一般物体での CAD 精度検証
- [ ] Qwen3-VL: 実画像でのグラウンディング精度改善 (vLLM 経由 8B)
- [ ] ROS2 Humble インストール (ネイティブ)
- [ ] YOLO26 → SAM3 → FoundationPose パイプライン統合
- [ ] ROS2 ノード実装

---

## 参照ドキュメント

| ドキュメント | 内容 |
|------------|------|
| `results/comparison_report.md` | YOLO26 / DINOv3 / FP / Qwen3-VL ベンチマーク詳細 |
| `results/glassboard_experiment_report.md` | ガラス板実験 Phase 1〜7 完全レポート |
| `results/3glass_experiment_report.md` | 透明グラス (容器) 実験レポート |
| `docs/project_glassboard_experiment.md` | ガラス板実験 進捗サマリ |
| `docs/exp2/plan_exp2.md` | 第2段実験 計画書 |
| `docs/exp2/handoff_exp2.md` | 第2段実験 ハンドオフメモ |
| `log/session_*.md` | 各セッションの作業ログ |
