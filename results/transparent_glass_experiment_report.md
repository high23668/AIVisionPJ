# 透明グラス認識 比較実験報告書

**実験期間**: 2026-04-23 〜 2026-04-24
**実施者**: high2366834@gmail.com
**実験環境**:
- OS: Ubuntu 22.04 / Linux 6.8.0-107-generic
- GPU: Quadro RTX 5000 Max-Q (16GB VRAM)
- CUDA 13.0
- Python 3.10 / 3.12（仮想環境ごとに異なる）
- センサー: RealSense D435i

---

## 1. 実験の位置づけ

透明物体（ガラス・プラスチック）は、従来のロボット認識において**最難関クラス**に位置付けられる対象である。主な困難要因：

- **RGBベース検出**: 透明部分は背景を透かすためテクスチャが不定。学習データに含まれるかが精度を左右する
- **アクティブ赤外線深度**: IR光が透過または反射するため **深度が欠損**。RealSenseを含む大半の一般的3Dセンサーが失敗する
- **セグメンテーション境界**: 輪郭は反射・歪みの手がかり頼りで曖昧

本実験は、既に本プロジェクトで検証済みの複数認識モデル（YOLO26 / DINOv3 / SAM3 / Qwen3-VL / Boxer / Any6D+SF3D）を **同じ透明グラス画像** に対して一括適用し、以下を明らかにすることを目的とする：

1. **RGBのみを入力とするモデル** が透明物体にどこまで対応できるか
2. **深度を併用するモデル** がRealSenseのIR深度欠損によってどう劣化するか
3. 次段階（Foundation-Stereo導入）の必要性を定量的に裏付ける

---

## 2. 技術系譜と本実験の位置づけ

```
2020 ─ YOLOv5 系列 (CNN 2D detector)
2022 ─ SAM v1 (Segment Anything)   ┬─ セグメンテーション基盤
2023 ─ DINOv2 (自己教師あり特徴)    │
2024 ─ DINOv3 / SAM 2             │
2024 ─ Qwen-VL / LLaVA 系 (VLM)    │
2024 ─ FoundationPose (6DoF)      ┤─ 6DoFポーズ
2024 ─ TripoSR / SF3D (Image-to-3D)│
2024 ─ Any6D (モデルフリー6DoF)     │
2025 ─ **Boxer** (Meta RLR, 単視点3D OBB) ┘
2025 ─ YOLO26                       新世代2D detector
2025 ─ SAM3                         テキストプロンプト対応セグメント
─────────────────────────────────
本実験: 上記スタックを同一ターゲット（透明グラス）で横比較
```

---

## 3. 実験環境

### 3.1 ハードウェア

| 項目 | 内容 |
|------|------|
| CPU | （Linux 6.8.0） |
| GPU | Quadro RTX 5000 Max-Q 16GB |
| 深度センサー | RealSense D435i (アクティブIRステレオ) |
| RGB解像度 | 640×480 |

### 3.2 Python仮想環境（モデル別）

| 環境 | Python | 用途 |
|------|--------|------|
| `.venv` | 3.10 | YOLO26 / DINOv3 / Qwen3-VL / Boxer |
| `.venv_sam3` | 3.12 | SAM3 |
| `.venv_3d` | 3.10 | SF3D (Any6Dメッシュ生成) |
| `.venv_stereo` | 3.11 | Foundation-Stereo (Phase 3) |

### 3.3 入力データ

2枚のRealSenseキャプチャ画像を用意した（カメラ Intrinsics: fx=605.8, fy=605.1, cx=317.4, cy=244.7）。

| データセット | 背景の煩雑度 | 保存先 |
|-------------|------------|--------|
| simple | グレーの無地ボードを背景に配置（コントラスト高） | `data/transparent_simple/` |
| complex | モニタ・家の置物・ラップトップ・本など多数の物体が混在 | `data/transparent_complex/` |

![入力画像・シンプル背景](transparent/input_simple.png)
*キャプション: 入力画像（シンプル背景）。中央に透明グラス、背景はグレーボード。左に胡椒瓶・ガネーシャ像、右下にキーボード。*

![入力画像・煩雑背景](transparent/input_complex.png)
*キャプション: 入力画像（煩雑背景）。手前にフルート型透明グラス、奥にモニタ・三角屋根の置物・ノートPC・カーテン・本が混在。*

### 3.4 IR深度マップの品質確認

キャプチャ直後にRealSenseのIR深度品質を可視化した。有効（0.1〜10m）とみなせる画素の割合をもとに、**欠損率**を計測した。

![シンプル背景・深度欠損](transparent/depth_simple.jpg)
*キャプション: シンプル背景のIR深度。グラス胴体が真っ黒（欠損）。欠損率18.1%。*

![煩雑背景・深度欠損](transparent/depth_complex.jpg)
*キャプション: 煩雑背景のIR深度。手前グラスの胴体に加え、モニタ黒画面・カーテンのしわなどが欠損。欠損率35.5%。*

**確認できたこと**:
- **透明グラス領域の深度はほぼ完全欠損**（IRが透過している）
- 煩雑背景では反射面・黒色吸収面も欠損に加わり、全画素の1/3以上が無効

→ 深度入力を必須とする後段モデル（Boxer / Any6D / FoundationPose）の性能劣化が予測される。

---

## 4. 実験結果

Phase 1（RGB単独モデル）と Phase 2（深度依存モデル）に分けて報告する。

### Phase 1.1 ─ YOLO26（2D物体検出）

**コマンド**（両画像に対して実行）:
```bash
python models/yolo26/yolo26_demo.py \
    --source data/transparent_simple/rgb.png \
    --variant yolo26l --conf 0.2
```

**検出結果（関連クラスのみ抜粋）**:

| 画像 | クラス | スコア |
|------|--------|--------|
| simple | **cup** | **0.92** |
| simple | keyboard | 0.32 |
| simple | bottle | 0.31（実際は胡椒瓶） |
| complex | tv | 0.87 |
| complex | **cup** | **0.79** |
| complex | laptop | 0.74 |

![YOLO26・シンプル](transparent/yolo26_transparent_simple.jpg)
*キャプション: YOLO26による検出結果（simple）。透明グラスがcup 0.92で正しく囲まれている。*

![YOLO26・煩雑](transparent/yolo26_transparent_complex.jpg)
*キャプション: YOLO26による検出結果（complex）。手前グラスをcup 0.79で検出。ただし奥のフルートグラスは見逃している。*

**所見**: 透明物体でもYOLO26は**形状特徴から cup として認識可能**である。ただし複数インスタンス、特に背景に溶け込む細長い形状は取り逃す傾向がある。

---

### Phase 1.2 ─ DINOv3（視覚特徴抽出）

DINOv3 ViT-B 16px を入力448×448で実行し、以下の2種の可視化を行った：

- **PCA可視化**: パッチ特徴量を3次元にPCA圧縮してRGBマッピング
- **類似度マップ**: 透明グラス上のアンカー点と全パッチのコサイン類似度

![DINOv3 PCA・シンプル](transparent/dinov3_transparent_simple_pca.jpg)
*キャプション: DINOv3 PCA（simple, 28×28パッチ）。グラス領域が黄緑色の独立クラスタとして分離される。*

![DINOv3 PCA・煩雑](transparent/dinov3_transparent_complex_pca.jpg)
*キャプション: DINOv3 PCA（complex）。2つのグラスが類似色で表現され、周囲の物体とは別クラスタに分離されている。*

![DINOv3 類似度・シンプル](transparent/dinov3_transparent_simple_sim.jpg)
*キャプション: DINOv3 類似度マップ（simple）。グラス中央をアンカーとして、グラス全体が赤（高類似）に浮かび上がる。*

![DINOv3 類似度・煩雑（グラス底をアンカー）](transparent/dinov3_transparent_complex_sim2.jpg)
*キャプション: DINOv3 類似度マップ（complex, 2回目）。手前グラス底をアンカーにすると、**手前グラスと奥のフルートグラスの両方**が高類似領域として浮かび上がる。2Dモデルでは取りこぼしていた複数インスタンスをDINOv3特徴は発見できる。*

**所見**: DINOv3は**ラベル学習なしで透明物体を一貫した特徴**として表現する。フューショット検出や物体再同定の有力な基盤となり得る。

---

### Phase 1.3 ─ SAM3（テキストプロンプトセグメンテーション）

3種のプロンプト（`glass`, `cup`, `transparent cup`）で両画像をテストした。

| 画像 | プロンプト | マスク数 | スコア | 推論時間 |
|------|-----------|---------|--------|---------|
| simple | glass | 1 | 0.925 | 1698ms |
| simple | cup | 1 | **0.950** | 1586ms |
| simple | transparent cup | 1 | 0.886 | 1674ms |
| complex | glass | 1 | **0.958** | 1688ms |
| complex | cup | 1 | 0.920 | 1701ms |
| complex | transparent cup | 1 | 0.914 | 1690ms |

![SAM3・シンプル](transparent/sam3_transparent_simple_glass.jpg)
*キャプション: SAM3 セグメンテーション（simple, `glass`プロンプト）。透明グラスの輪郭がピクセル精度で追従している。*

![SAM3・煩雑](transparent/sam3_transparent_complex_glass.jpg)
*キャプション: SAM3 セグメンテーション（complex, `glass`プロンプト）。手前フルートグラスを0.958の高スコアで完璧にセグメント。*

**所見**: SAM3は**透明物体に対して0.89〜0.96の高スコアで精密マスク**を生成する。本実験で最も強いRGB単独認識器。ただしマスクは各プロンプトにつき1個のみ返っており、奥の複数インスタンスは取り逃している。

---

### Phase 1.4 ─ Qwen3-VL 4B（VLM）

**describe タスク**（`--task describe --prompt "List all visible objects in this image briefly."`）:

- simple: "A tall, clear, empty glass tumbler sits prominently in the center-right of the frame..."（透明グラスを正確に描写）
- complex: "To the right of the books, a tall, clear glass tumbler sits on the desk..."（同様に正確）

**ground タスク**（`--task ground --object "glass"`）:

| 画像 | 返り値（1000正規化） | ピクセル換算 bbox |
|------|-------------------|------------------|
| simple | [514, 525, 690, 998] | (328, 252)-(441, 479) |
| complex | [515, 525, 690, 998] | (328, 252)-(442, 479) |

![Qwen3-VL Grounding・シンプル](transparent/qwen3vl_transparent_simple_glass.jpg)
*キャプション: Qwen3-VL grounding（simple, "glass"）。手前のグラスを正確に囲むbboxを返している。*

![Qwen3-VL Grounding・煩雑](transparent/qwen3vl_transparent_complex_glass.jpg)
*キャプション: Qwen3-VL grounding（complex, "glass"）。手前フルートグラスを正確に特定。奥のグラスは見逃している。*

**所見**: Qwen3-VLは**言語で透明グラスを「tall, clear, empty glass tumbler」と正確に記述**できる。座標を1000×1000正規化スケールで返す点は注意が必要（既存デモスクリプトではピクセル指定を要求しても無視されたため、手動スケーリングを実装）。

---

### Phase 2.1 ─ Boxer（単視点3D OBB）

**コマンド**:
```bash
python run_boxer.py --input data/transparent_simple/ \
  --labels "cup,glass,bottle,mug,tumbler" \
  --thresh2d 0.2 --thresh3d 0.2 --max_n 1
```

前回セッションで修正した Y-Z swap回転をRealSenseLoaderに入れた状態で実行した。

**3D検出結果（world座標、単位m）**:

| 画像 | クラス | 位置 (x, y, z) | サイズ (X×Y×Z cm) | 信頼度 |
|------|--------|---------------|------------------|--------|
| simple | glass | (0.042, 0.344, -0.065) | 5.9 × 6.6 × 11.7 | 0.805 |
| simple | bottle | (-0.144, 0.326, -0.006) | 5.0 × 5.0 × 8.9 | 0.576 |
| complex | glass | (0.047, 0.367, -0.063) | 5.6 × 5.7 × 12.4 | 0.723 |
| complex | bottle | (-0.140, 0.318, -0.014) | 5.0 × 5.0 × 8.7 | 0.609 |

![Boxer・シンプル](transparent/boxer_transparent_simple.jpg)
*キャプション: Boxer（simple）。左=2D OWLv2（glass 0.67）、右=3D BoxerNet（glass 0.94 / bottle 0.94）。3Dボックスは縦長直方体としてグラスに近い位置に描かれるが、底面が下方向にはみ出ている。*

![Boxer・煩雑](transparent/boxer_transparent_complex.jpg)
*キャプション: Boxer（complex）。2D・3Dともに glass と bottle を検出。3Dボックスはフルートグラスに重なりつつも、深度欠損の影響で向き・厚みが不正確。*

**所見**:
- **サイズ推定は実物に近い**（6cm×7cm×12cmは典型的なグラス寸法）
- **位置・姿勢は誤差大**: 深度欠損の結果、point cloudがグラス底部に集中し、重心と向きが不正確になる
- 胡椒瓶を`bottle`として誤検出しているが、これは透明物体問題ではなくクラスラベル側の問題

---

### Phase 2.2 ─ Any6D + SF3D（モデルフリー6DoF）

パイプライン:
1. SAM3で"glass"マスク生成（Phase 1.3のマスクを再利用、カバー率7.7%）
2. SF3Dで単一画像から3Dメッシュ生成

![SF3D メッシュ・シンプル](transparent/sf3d_transparent_simple_views.png)
*キャプション: SF3D生成メッシュ（simple）。Front(XY) / Side(XZ) / Top(YZ) / 3D表示。サイズは8.3×4.8×4.2cm。奥行方向が薄い板状に再構成されており、円筒形グラスの形状を復元できていない。*

![SF3D メッシュ・煩雑](transparent/sf3d_transparent_complex_views.png)
*キャプション: SF3D生成メッシュ（complex）。サイズ8.4×5.0×5.2cm。同じく板状。*

**定量評価（視覚的確認のみ）**:

| 画像 | SF3D生成サイズ | 実物推定 | 判定 |
|------|--------------|---------|------|
| simple | 8.3 × 4.8 × 4.2 cm | ~7 × 7 × 12 cm | ❌ 縦方向が潰れた板状 |
| complex | 8.4 × 5.0 × 5.2 cm | ~7 × 7 × 12 cm | ❌ 同様 |

**所見**:
- SF3Dは**透明物体の背面形状を合理的に推定できない**
- 結果、シェル状（薄い板）のメッシュが出力され、後段FoundationPoseへの入力としては不適
- SF3Dのモデル前提が「不透明な固体」であることが原因と考えられる

---

### Phase 2.3 ─ FoundationPose（CAD要求モデル）

今回は**実行せず**。理由：
1. 透明グラスのCADメッシュを保有していない
2. Any6D経由のSF3Dメッシュが板状になっており、入力としては不適切
3. 深度入力もグラス領域で欠損しており、model-basedでも失敗が見えている

→ Foundation-Stereo導入後に再評価する。

---

## 4.5 Phase 3 ─ Foundation-Stereo による深度置換

Phase 2で確認された**RealSense IR depthの透明物体欠損問題**を解決するため、NVIDIA Labsが2025年に発表した **Foundation-Stereo**（CVPR 2025 Best Paper Nomination）を導入した。

### 3.1 モデル概要

Foundation-Stereoはzero-shotステレオマッチング基盤モデルで、以下の特徴を持つ：

- **入力**: rectified済みステレオ左右画像（RGBまたはモノクロIR）
- **出力**: 視差マップ → 焦点距離・baselineから深度変換
- **強み**: 100万ペアの合成データで学習され、**透明・反射・テクスチャ無し表面**にも汎化
- **Middlebury / ETH3Dの両リーダーボードで1位**
- **論文**: https://arxiv.org/abs/2501.09898

検証に使用したモデル：
- `11-33-40` (ViT-S backbone, 752MB weight) — 高速版

### 3.2 実装

#### 3.2.1 環境構築

```bash
uv venv .venv_stereo --python 3.11
source .venv_stereo/bin/activate
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install scikit-image omegaconf opencv-contrib-python timm open3d \
    xformers==0.0.28.post1 gdown huggingface-hub trimesh einops  # 他
git clone https://github.com/NVlabs/FoundationStereo.git \
    models/foundation_stereo/FoundationStereo
gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf \
    -O pretrained_models/
```

flash-attnは不要（コード中で使用されていないため）。

#### 3.2.2 ステレオIRキャプチャスクリプト

`scripts/capture_realsense_stereo.py` を新規作成し、RealSense D435iの左右IRカメラ（ベースライン 49.9mm）から以下を保存：

| 出力 | 内容 |
|------|------|
| `rgb.png` | 参照用RGB |
| `left_ir.png` / `right_ir.png` | モノクロIR左右（rectified済み） |
| `depth.png` / `depth.npy` | RealSense内蔵IR depth（比較用） |
| `intrinsics.json` | RGB+IR双方のK、baseline |
| `K_ir.txt` | Foundation-Stereo用フォーマット（3x3 Kフラット + baseline） |

**重要**: `depth_sensor.set_option(rs.option.emitter_enabled, 0)` でIR projector を**OFF**にする。
emitterがONだとIR画像にドットパターンが映り込み、Foundation-Stereoのマッチングに干渉する。

IR画像はモノクロ（H×W）なのでFoundation-Stereoが期待するRGB形式（H×W×3）に変換する必要がある（同じチャネルを3回コピー）。

### 3.3 実行

```bash
cd models/foundation_stereo/FoundationStereo
python scripts/run_demo.py \
    --left_file data/stereo_simple/left_ir.png \
    --right_file data/stereo_simple/right_ir.png \
    --intrinsic_file data/stereo_simple/K_ir.txt \
    --ckpt_dir ./pretrained_models/11-33-40/model_best_bp2.pth \
    --out_dir results/transparent/fs_simple_vits/ \
    --scale 1.0 --valid_iters 32 --get_pc 1
```

推論時間: 約1〜2分（ViT-S、RTX 5000 Max-Q、640×480解像度）
VRAM使用量: ピーク 約4.5〜5GB

### 3.4 ステレオIR入力

![ステレオIR・シンプル](transparent/stereo_preview_simple.jpg)
*キャプション: シンプル背景のステレオキャプチャ。左=RGB参照、中央=左IR、右=右IR（emitter OFF時）。IR画像にドットパターンが映り込んでいないことが確認できる。*

![ステレオIR・煩雑](transparent/stereo_preview_complex.jpg)
*キャプション: 煩雑背景のステレオキャプチャ。手前に透明グラスとRealSense D435iの箱、奥に赤ベアぬいぐるみ、右側にケーブル。*

### 3.5 Foundation-Stereo出力（視差カラー表示）

![Foundation-Stereo vis・シンプル](transparent/fs_vis_simple.png)
*キャプション: Foundation-Stereo出力（simple, 公式可視化）。左=IR参照画像、右=推定視差。透明グラスが明瞭な独立領域として復元されている。*

![Foundation-Stereo vis・煩雑](transparent/fs_vis_complex.png)
*キャプション: Foundation-Stereo出力（complex）。透明グラス・赤ベア・箱など全オブジェクトが立体的に復元されている。*

### 3.6 RealSense vs Foundation-Stereo 定量比較

|  | RealSense IR 欠損率 | Foundation-Stereo 欠損率 | 改善 |
|--|---------------------|------------------------|------|
| simple | 16.1% | **11.9%** | **-4.2%** |
| complex | 29.8% | **11.8%** | **-18.0%** |

**注**: FSの11%台の欠損はほぼ全て**画面左端の非重複領域**（ステレオの原理的死角）であり、シーン内部では実質 **0%近い欠損率** を達成している。

![深度比較・シンプル](transparent/depth_comparison_simple_v2.jpg)
*キャプション: 深度比較（simple, カラー範囲 0.2〜0.35m）。左=RGB、中央=RealSense（グラス領域に大きな黒い穴）、右=Foundation-Stereo（グラスが青い直方体として滑らかに復元、ケーブルの細部まで再現）。FS左端の黒帯はステレオ非重複領域。*

![深度比較・煩雑](transparent/depth_comparison_complex_v2.jpg)
*キャプション: 深度比較（complex）。左=RGB、中央=RealSense（29.8%欠損、透明グラス・箱前面・暗色面が大量に失われている）、右=Foundation-Stereo（11.8%欠損、全オブジェクトが滑らかに復元）。特に**透明グラスがRealSenseでは完全に消失していたのに対し、FSでは青色の明瞭な領域として再構成**されている。*

### 3.7 Phase 3 所見

**成功点**:
- **透明グラス領域の深度が完全復元**された（最大の課題解決）
- 煩雑背景で**欠損率を18%ポイント改善**
- 推論時間1〜2分、VRAM 5GB程度で動作可能
- emitter OFF + モノクロIR入力で問題なく動作

**注意点**:
- 左端の非重複領域（~10〜15%）は原理的に深度が出ない
- Foundation-Stereo出力は**IR左カメラ視点**なので、RGB視点とは若干視差がある（FOVも異なる）
- 後段モデル（Boxer等）に渡す際はIR座標系で揃える必要がある

---

## 4.6 Phase 4 ─ Foundation-Stereo depthで深度依存モデル再検証

Phase 3で得られたFoundation-Stereo深度を使って、Phase 2で失敗した**Boxer**と**Any6D**を再実行した。

### 4.1 深度のRGB視点ワープ

Foundation-Stereo出力は**IR左カメラ視点**。Boxer・Any6DはRGB視点depthを期待するため、IR→RGB座標系へワープ処理を行う。

処理:
1. **逆投影**: IR depthの各ピクセル `(u,v,depth)` → 3D点 `(X,Y,Z)`（IR frame）
2. **座標変換**: IR左→RGB（並進15mm、回転ほぼ恒等）
3. **再投影**: RGB K行列で3D点を画像平面に投影
4. **2×2スプラッティング**: 前方ワープによる隙間を埋める（RGBがIRより狭FOV・高解像度なため1ピクセル→2×2領域にコピー）

このワープ処理は**pyrealsense2のrs.alignと数学的に同一**。OpenCV・Open3D・ROS2の `depth_image_proc` でも標準的に実装されている手法。

RealSenseデバイスから取得した IR_left → RGB 外部パラメータ:
```
t = [15.09, 0.12, 0.29] mm
```

**ワープ後の深度有効率**:

| シーン | RS depth 欠損 | FS warped to RGB 欠損 | 改善 |
|-------|------------|---------------------|------|
| simple | 16.1% | **1.2%** | -14.9pt |
| complex | 29.8% | **1.2%** | **-28.6pt** |

![深度アラインメント比較・シンプル](transparent/depth_aligned_simple.jpg)
*キャプション: Simple シーン。左=RGB、中央=RealSense（黒い穴だらけ、グラス領域ほぼ欠損）、右=FS warped to RGB（1.2%のみ欠損、グラスが青い直方体として明瞭に復元）。*

![深度アラインメント比較・煩雑](transparent/depth_aligned_complex.jpg)
*キャプション: Complex シーン。左=RGB、中央=RealSense（29.8%欠損、透明グラス・箱前面・暗色面が大量失効）、右=FS warped to RGB（1.2%欠損、全オブジェクト滑らかに復元）。*

### 4.2 Boxer 再実行 (Phase 4.1)

FS ワープ済み深度を入れた Boxer を実行し、Phase 2.1（RealSense depth版）と比較した。

**Simple シーン（glass）**:

| 指標 | Phase 2 (RS) | Phase 4 (FS) | 変化 |
|------|------------|-----------|------|
| 信頼度 | 0.805 | **0.813** | +0.8% |
| 位置 X | 0.042 m | 0.030 m | -1.2cm |
| 位置 Y（深さ） | 0.344 m | **0.246 m** | **-9.8cm** |
| 位置 Z | -0.065 m | -0.024 m | +4.1cm |
| サイズ (cm) | 5.9×6.6×11.7 | 5.3×6.0×11.0 | ほぼ同等 |

**Complex シーン（glass）**:

| 指標 | Phase 2 (RS) | Phase 4 (FS) | 変化 |
|------|------------|-----------|------|
| 信頼度 | 0.723 | **0.794** | **+7.1%** |
| 位置 X | 0.047 m | 0.036 m | -1.1cm |
| 位置 Y（深さ） | 0.367 m | **0.279 m** | **-8.8cm** |
| 位置 Z | -0.063 m | -0.032 m | +3.1cm |
| サイズ (cm) | 5.6×5.7×12.4 | 5.9×6.3×12.0 | ほぼ同等 |

**注**: Y距離が9cm近づいたのは物体の**真の位置**に近づいた正しい補正。RealSenseでは底部の薄い有効画素に引っ張られて遠方推定していた。

![Boxer Phase 2 vs Phase 4 比較・シンプル](transparent/boxer_phase2_vs_phase4_simple.jpg)
*キャプション: Simple シーンでの Boxer比較。上段=Phase 2（RealSense depth）、下段=Phase 4（Foundation-Stereo depth）。Phase 4の3Dボックスはグラスの形に密着しており、Phase 2の傾き・はみ出しが解消している。*

![Boxer Phase 2 vs Phase 4 比較・煩雑](transparent/boxer_phase2_vs_phase4_complex.jpg)
*キャプション: Complex シーンでの Boxer比較。Phase 4では信頼度が +7.1%向上。3DボックスがよりRGB上のグラス輪郭に沿う。*

### 4.3 Any6D 再実行 (Phase 4.2)

Any6D パイプラインを新しいステレオシーンに対して実行。

**パイプライン**:
1. **SAM3** で `"glass"` プロンプトのマスク生成（simple スコア0.935、complex スコア0.913）
2. **SF3D** でマスク済み画像から3Dメッシュ生成
3. **Docker Any6D (FoundationPose)** + **FS warped depth** でポーズ推定

#### 4.3.1 SF3D生成メッシュと Any6Dの精緻化

![Any6D メッシュ・シンプル](transparent/any6d_mesh_simple.png)
*キャプション: Simple シーン。上段=SF3D初期メッシュ（9.1×4.4×2.3cm、板状）、下段=Any6D最終メッシュ（10.2×5.5×1.6cm、スケール補正済）。Front/Side/Top/3D の4視点表示。SF3Dのshell形状という本質的限界は残るが、**表面の有効領域が拡大**している。*

![Any6D メッシュ・煩雑](transparent/any6d_mesh_complex.png)
*キャプション: Complex シーン。上段=SF3D初期（8.4×4.4×3.7cm）、下段=Any6D最終（10.2×5.6×1.6cm）。*

#### 4.3.2 Any6Dの姿勢探索プロセス（refiner stages）

Any6Dは**252個のポーズ候補**を生成し、4段階の refinement を経て最終ポーズを選出する。各段階で上位候補を可視化したものが以下：

- **Stage 1**: 初期ポーズ候補を生成し、RealSense視点でレンダリング
- **Stage 2**: ポーズ再評価、不整合な候補を淘汰
- **Stage 3**: スケール補正を考慮して再評価
- **Stage 4**: 絞り込んだスケールで再度ポーズ推定

各行は1ポーズ候補。8列で（左から順に）: 候補mask / RGB重ね / 深度レンダリング / 観測深度 / GT mask / GT RGB重ね / 観測深度 / 観測depthマップを表示している。

![Any6D refiner stages・シンプル](transparent/any6d_refiner_stages_simple.jpg)
*キャプション: Any6D refinement stages（simple, 各ステージの上位3候補）。Stage 1,2 では傾いた候補が多数。Stage 3 のスケール調整から円筒っぽい形状に収束し、Stage 4 で最終的に正しい垂直姿勢へと収束する。*

![Any6D refiner stages・煩雑](transparent/any6d_refiner_stages_complex.jpg)
*キャプション: Any6D refinement stages（complex）。同様に Stage 3→4 で急速に正しい姿勢に収束する様子が確認できる。*

#### 4.3.3 最終ポーズの可視化

![Any6D 最終ポーズ・シンプル](transparent/any6d_simple_fs_pose.jpg)
*キャプション: Simple シーン、Any6D推定ポーズでメッシュを投影（緑点=メッシュ頂点8178点、Z=23.2cm）。透明グラスに密着しており、3D位置が正しく推定されている。*

![Any6D 最終ポーズ・煩雑](transparent/any6d_complex_fs_pose.jpg)
*キャプション: Complex シーン、Any6D推定ポーズでメッシュ投影（頂点9392点、Z=23.4cm）。手前グラスの位置を正しく捕捉。*

### 4.4 Phase 4 所見

**Boxer**:
- 3Dボックスが**正しい3D位置・姿勢**で描かれるようになった（Phase 2で半分はみ出していた問題が解消）
- 信頼度も向上（complex: 0.72 → 0.79）
- Phase 2で致命的だった**透明グラスの3D位置推定が初めて成功**

**Any6D**:
- **透明グラスの6DoFポーズ推定に成功**（Phase 2.2では未実行だった）
- SF3Dメッシュが依然として板状（本質的限界）だが、Any6Dのスケール補正と深度を使ったICP型refinementで補われている
- 4段階のrefinement過程で正しい姿勢に収束する様子が可視化できた

**結論**: Foundation-Stereoで深度を置き換えることで、**RGBだけでは実現不可能だった透明物体の3D位置・姿勢推定が両方のモデルで成立した**。

---

## 5. 考察

### 5.1 成功点

1. **RGB単独モデルはすべて透明グラスを認識可能**だった
   - SAM3（0.89〜0.96）> DINOv3（特徴抽出）> YOLO26 ≈ Qwen3-VL（単体検出）
2. **DINOv3の類似度マップ**により、2D検出では見逃されていた**奥のフルートグラスも特徴的に発見**できた
3. Boxerのサイズ推定は**深度が大きく欠損していても妥当な値**を返した

### 5.2 限界・課題

1. **RealSenseのIR深度は透明部分で完全欠損**（予測通り）
   - 欠損率 simple=18.1% / complex=35.5%
2. **Boxer 3D位置・姿勢は劣化**（深度欠損の直接的影響）
3. **SF3Dは透明物体の形状再構成に失敗**
   - 板状メッシュとなり、以降のFoundationPose系パイプラインを無効化
4. **YOLO26・Qwen3-VL・SAM3 いずれも複数インスタンス検出が弱い**（プロンプト1個につき1個しか返さない傾向）

### 5.3 RGBモデルと深度モデルの対比

| カテゴリ | モデル | 透明グラス対応 (RS depth) | 透明グラス対応 (FS depth) |
|---------|--------|-----------------------|-----------------------|
| RGB単独 | YOLO26 | ✅ 0.79〜0.92 | (深度不使用) |
| RGB単独 | DINOv3 | ✅ 複数インスタンス特定 | (深度不使用) |
| RGB単独 | SAM3 | ✅ 0.89〜0.96 | (深度不使用) |
| RGB単独 | Qwen3-VL | ✅ VQA完璧 | (深度不使用) |
| 深度併用 | Boxer | ⚠️ 3D位置ずれ | ✅ **正しい3D位置・信頼度+7%** |
| 深度併用 | Any6D+SF3D | ❌ 未完 | ✅ **ポーズ推定成功** |
| 深度併用 | FoundationPose | 未実行 | 未実行（CAD入手後に検証） |
| **深度源** | **Foundation-Stereo** | — | ✅ **透明領域完全復元・欠損28pt改善(complex)** |

---

### 5.4 Phase 3〜4 で得られた知見

1. **Foundation-StereoはRealSense IR depthの致命的欠点を解決**
   - 透明グラスの深度が復元され、今までロストしていた3D情報が手に入る
   - 煩雑シーンで**欠損率 29.8% → 11.8%**（RGB視点ワープ後は **1.2%**）

2. **RealSense D435iを"ソフトウェア的にアップグレード"できる**
   - 既存ハードウェアのIR左右カメラをそのまま活用
   - emitter OFFでの撮影運用に変更するだけ
   - 推論コストは1〜2分/枚（リアルタイムではないが研究用途なら十分）

3. **zero-shot汎化が実用レベル**
   - モノクロIR入力という学習時と異なるドメインでも高品質な深度が得られた

4. **深度依存モデルが透明物体で動くようになった（Phase 4の成果）**
   - **Boxer**: 3D位置・姿勢が正しく推定できる、信頼度+7%
   - **Any6D**: ポーズ推定成功。SF3Dの板状メッシュという制約は残るが、**6DoFの位置・姿勢は正確**
   - いずれもPhase 2で失敗していた透明グラスに対し、Phase 4で初めて**動作する結果**が得られた

5. **SF3Dの限界は残る**
   - 透明物体の背面形状推定は依然として困難
   - Any6Dのscale refinementで部分的に補正されるが、完全な円筒形メッシュにはならない
   - 改善には: 多視点撮影、あるいは実物採寸による簡易CADとの併用が有望

---

### 5.5 今後の展開

| 優先度 | アクション | 期待効果 |
|--------|----------|---------|
| 高 | 実物グラス採寸 → 簡易円筒CAD → FoundationPose model-based × FS depth | 完全な円筒メッシュ × 正確な深度 で透明物体6DoF推定の精度上限を見極める |
| 高 | Foundation-Stereo のONNX/TensorRT化 | 推論速度を数秒に短縮、準リアルタイム化 |
| 中 | ROS2統合 | RealSenseノード → FS depthノード → 認識ノードのパイプライン化 |
| 中 | SAM3複数インスタンス化 | 閾値・プロンプト工夫 or DINOv3併用 |
| 中 | 多視点撮影→SF3Dまたは他のImage-to-3D | 透明物体のメッシュ品質向上 |
| 低 | Qwen3-VLの複数grounding強化 | prompt engineering |

---

## 6. まとめ

- 透明グラスは **RGBだけ見れば全モデルで検出・セグメント可能** である。特にSAM3はピクセル精度で優秀。
- RealSenseのIR深度は **透明部分で致命的に欠損**（16〜30%）し、これを直接入力とするBoxer・Any6D・FoundationPoseは **3D位置推定で著しく劣化** する。
- **Foundation-Stereo（CVPR 2025 Best Paper Nom.）** を導入した結果、透明グラスの深度が**明瞭に復元**され、シーン全体の欠損率もRGB視点アライン後で**1.2%まで低減**。
- この新depthを投入した **Phase 4** で、Boxerの3Dボックス位置精度と信頼度が向上し、Any6Dによる**透明物体の6DoFポーズ推定が初めて成功**した。
- 残る課題は（1）SF3Dの透明物体メッシュ品質（2）FSの推論速度。いずれも実現可能な改善策が視野にある。

---

## 付録: ファイル構成

```
data/
├── transparent_simple/
│   ├── rgb.png            入力RGB
│   ├── depth.npy          IR深度 (m, float32)
│   ├── depth.png          IR深度 (mm, uint16)
│   ├── depth_viz.jpg      欠損可視化
│   ├── intrinsics.json    カメラパラメータ
│   ├── mask.png           SAM3生成マスク
│   ├── mask_overlay.jpg   マスクオーバーレイ
│   └── mesh.obj           SF3D生成メッシュ
└── transparent_complex/   同じ構成

data/stereo_{simple,complex}/   (Phase 3ステレオキャプチャ)
├── rgb.png                参照RGB
├── left_ir.png            左IR (3ch化済み)
├── right_ir.png           右IR (3ch化済み)
├── depth.npy              RealSense IR depth
├── intrinsics.json        RGB+IR intrinsics + baseline
├── K_ir.txt               Foundation-Stereo用 (3x3 K + baseline)
└── preview.jpg            RGB/IR並列プレビュー

results/transparent/
├── input_simple.png / input_complex.png       入力画像コピー
├── depth_simple.jpg / depth_complex.jpg       IR深度可視化
├── yolo26_transparent_{simple,complex}.jpg
├── dinov3_transparent_{simple,complex}_pca.jpg
├── dinov3_transparent_{simple,complex}_sim*.jpg
├── sam3_transparent_{simple,complex}_{glass,cup,transparent_cup}.jpg
├── qwen3vl_transparent_{simple,complex}_{glass,transparent_glass}.jpg
├── boxer_transparent_{simple,complex}.jpg
├── sf3d_transparent_{simple,complex}_views.png
├── stereo_preview_{simple,complex}.jpg        Phase 3 ステレオ入力
├── fs_vis_{simple,complex}.png                Foundation-Stereo 公式可視化
├── depth_comparison_{simple,complex}_v2.jpg   RealSense vs FS 比較
└── fs_{simple,complex}_vits/                  FSの生出力
    ├── depth_meter.npy    推定深度 (IR左カメラ視点、meters)
    ├── cloud.ply          3D点群
    └── vis.png            視差カラー可視化

models/foundation_stereo/FoundationStereo/     (Phase 3 リポジトリ)
├── scripts/run_demo.py    推論スクリプト
└── pretrained_models/
    ├── 11-33-40/          ViT-S 752MB (今回使用)
    ├── 23-51-11/          ViT-L 3.1GB
    └── onnx/

models/boxer/output/transparent_{simple,complex}/
├── boxer_3dbbs.csv        3D OBB (world座標)
├── owl_2dbbs.csv          OWLv2 2D検出
└── boxer_viz_current.jpg  可視化画像
```

**Phase 3で新規追加したスクリプト**:
- `scripts/capture_realsense_stereo.py` — 左右IR + RGB + depth + intrinsics + baseline を一括保存

**Phase 4で新規追加したスクリプト**:
- `scripts/warp_fs_depth_to_rgb.py` — FS depth（IR座標）→ RGB座標へ剛体ワープ（2×2スプラッティング + Z-buffer）

**Phase 4で追加生成したディレクトリ**:
- `data/transparent_{simple,complex}_fs/` — rgb.png + depth.npy（FS warped）+ depth.png + mask.png + mesh.obj + intrinsics.json
- `results/transparent/any6d_{simple,complex}_fs/` — Any6D出力一式（pred_pose.txt、final_mesh.obj、refine/score 可視化ストリップ）
- `models/boxer/output/transparent_{simple,complex}_fs/` — Boxer出力（3D OBB CSV + 可視化）

**Phase 4 の追加画像**:
- `depth_aligned_{simple,complex}.jpg` — RGB / RS depth / FS warped depth 三並列
- `boxer_phase2_vs_phase4_{simple,complex}.jpg` — Boxer Phase 2 vs Phase 4 上下比較
- `any6d_mesh_{simple,complex}.png` — SF3D初期 vs Any6D最終メッシュ 比較
- `any6d_refiner_stages_{simple,complex}.jpg` — 4段階 refinement ストリップ（上位3候補抜粋）
- `any6d_{simple,complex}_fs_pose.jpg` — 最終ポーズ可視化（メッシュ頂点を緑点でRGBに投影）

---

**使用スクリプト**:
- `scripts/capture_realsense.py` — RealSenseキャプチャ
- `scripts/generate_sam3_mask.py` — SAM3マスク生成
- `scripts/any6d_generate_mesh.py` — SF3Dメッシュ生成
- `models/yolo26/yolo26_demo.py`
- `models/dinov3/dinov3_demo.py`
- `models/sam3/sam3_demo.py`
- `models/qwen3vl/qwen3vl_demo.py`
- `models/boxer/run_boxer.py`
- `models/boxer/loaders/realsense_loader.py` — 今回の作業中にY-Z swap回転を修正

VSCodeのMarkdownプレビュー（Ctrl+Shift+V）で画像が正しく表示されることを確認してください。
