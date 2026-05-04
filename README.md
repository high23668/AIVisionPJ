# AIVisionPJ - 使い方ガイド

ロボット認識パイプライン: YOLO26 / DINOv3 / SAM3 / FoundationPose / Qwen3-VL

---

## 仮想環境について

このプロジェクトには2つの仮想環境があります:

| 環境 | 用途 | アクティベート |
|------|------|--------------|
| `.venv` (Python 3.10) | YOLO26 / DINOv3 / FoundationPose / Qwen3-VL | `source /home/vr01/AIVisionPJ/.venv/bin/activate` |
| `.venv_sam3` (Python 3.12) | SAM3 専用 | `source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate` |

**YOLO26 / DINOv3 を使う場合 (毎回必須):**
```bash
source /home/vr01/AIVisionPJ/.venv/bin/activate
```

**SAM3 を使う場合:**
```bash
source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate
```

---

## カメラのsource番号について

RealSense D435I のカメラsource番号はPCの再起動や接続順で変わることがあります。
以下のコマンドで現在の番号を確認できます:

```bash
python -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'source={i}: {int(w)}x{int(h)} @ {fps:.0f}fps')
        cap.release()
" 2>/dev/null
```

**見方:**
- `320x180` や `424x240` が出ている番号 → RealSenseの深度/IR (使わない)
- `640x480 @ 90fps` → RealSenseのIR (赤外線 = 白い点々)
- `640x480 @ 30fps` が複数 → その中でカラー映像が出るものを探す

実際に映像を確認するには:
```bash
yolo detect predict model=yolo26n.pt source=<番号> show=True
```
カラー映像が出たらその番号を使う。(前回の確認: **source=4 がRealSenseカラー**)

---

## YOLO26 - リアルタイム物体検出

### リアルタイム検出ウィンドウを起動

```bash
# RealSenseカメラでリアルタイム検出
yolo detect predict model=yolo26n.pt source=4 show=True

# 信頼度しきい値を上げて誤検出を減らす (デフォルト0.25)
yolo detect predict model=yolo26n.pt source=4 show=True conf=0.5

# 特定クラスのみ表示 (例: person=0, bottle=39, cup=41)
yolo detect predict model=yolo26n.pt source=4 show=True classes=0,39,41
```

**終了: ウィンドウで `q` キー**

### COCO クラス番号の主なもの

| 番号 | クラス | 番号 | クラス |
|------|--------|------|--------|
| 0 | person | 39 | bottle |
| 14 | bird | 41 | cup |
| 15 | cat | 56 | chair |
| 16 | dog | 63 | laptop |
| 24 | backpack | 66 | keyboard |
| 26 | handbag | 67 | cell phone |

### モデルサイズ比較

| モデル | 速度 | 精度 |
|--------|------|------|
| yolo26n | 最速 (73 FPS) | 低め |
| yolo26s | 速い | 普通 |
| yolo26m | 普通 | 高め |

```bash
# モデルを変えるには model= を変更
yolo detect predict model=yolo26s.pt source=4 show=True
```

---

## DINOv3 - ゼロショット物体認識

**特徴**: アノテーション・学習なしで「この物体を覚えて認識して」ができる。

### Step 1: 物体を登録する

カメラに物体を向けてEnterを押すと登録されます。何個でも登録できます。

```bash
# コップを登録
python models/dinov3/dinov3_demo.py --task register --label cup --source 4

# 鉛筆を登録
python models/dinov3/dinov3_demo.py --task register --label pencil --source 4

# 本を登録
python models/dinov3/dinov3_demo.py --task register --label book --source 4
```

登録した参照画像は `results/ref_<ラベル名>.jpg` に保存されます。

### Step 2: ライブ認識を起動する

```bash
python models/dinov3/dinov3_demo.py --task recognize --source 4
```

カメラ映像が表示され、登録した物体との類似度がリアルタイムでオーバーレイされます。

- 緑のバー + 数値 = 最も近いと判断した物体
- スコアは 0.0〜1.0 (1.0に近いほど類似)
- **終了: ウィンドウで `q` キー**

### 登録をリセットしたい場合

```bash
rm -f results/gallery.npz results/ref_*.jpg
```

### 認識精度を上げるコツ

- 登録時は物体がフレームの中央に来るようにする
- 背景はシンプルな方が良い
- 同じ物体を複数アングルで登録すると認識が安定する (同じラベルで複数回登録すると上書きされるので、angle1/angle2 のようにラベルを分ける)

---

## SAM3 - テキストプロンプトでセグメンテーション

**特徴**: 「cup」「person」などテキストを入力するだけで、その物体のマスク(輪郭)を切り出す。
YOLOがBBoxを返すのに対し、SAM3はピクセル単位の正確な形状を返す。

**仮想環境**: `.venv_sam3` (Python 3.12) を使用

### 使い方

```bash
source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate

# カメラで1枚撮影してセグメント (結果はシステムビューアで自動表示)
DISPLAY=:0 python models/sam3/sam3_demo.py --task camera --source 4 --prompt "cup"

# ライブモード (プロンプト入力 → 撮影 → 結果表示 を繰り返す)
DISPLAY=:0 python models/sam3/sam3_demo.py --task live --source 4

# 静止画ファイルをセグメント
DISPLAY=:0 python models/sam3/sam3_demo.py --task image --image results/camera_capture.jpg --prompt "bottle"

# ストリーミングモード (~0.6FPS) → ブラウザで http://localhost:8080 を開く
DISPLAY=:0 python models/sam3/sam3_demo.py --task stream --source 4 --prompt "cup"
```

**注意**: VSCodeターミナルから実行する場合は `DISPLAY=:0` が必須。

結果は `results/sam3_*.jpg` に保存されます。

### タスク一覧

| タスク | 説明 | 表示方法 |
|--------|------|--------|
| `camera` | 1枚撮影してセグメント | システムビューアで表示 |
| `live` | プロンプト入力→撮影→表示を繰り返す | システムビューアで表示 |
| `image` | 静止画ファイルをセグメント | システムビューアで表示 |
| `stream` | 連続撮影+セグメント (~0.6FPS) | ブラウザ (MJPEG) |

### HuggingFaceアクセスについて

SAM3はMeta側の手動承認が必要 (承認済み: 2026-03-06)。
トークンが切れた場合は再ログイン:

```bash
source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate
python -c "from huggingface_hub import login; login(token='<your_token>')"
```

---

## トラブルシューティング

### カメラが真っ暗 / 真緑の画像が撮れる
カメラのauto-exposureが安定していません。コマンドを再実行してください。

### `source=X: cannot open` が出る
RealSenseが認識されていません。USB接続を確認して `rs-enumerate-devices` を実行してください。

### YOLO26モデルが見つからない
初回実行時に自動ダウンロードされます。ネット接続を確認してください。

### DINOv3ロード時に `GatedRepoError`
HuggingFaceの認証が切れています:
```bash
# .venv で実行
python -c "from huggingface_hub import login; login(token='<your_token>')"
```

### SAM3で `GatedRepoError` / `awaiting a review`
Meta側の手動承認待ちです。承認メールを待ってから `.venv_sam3` でHFログインしてください。

---

## FoundationPose + RealSense - 6DoF姿勢推定

**重要**: FoundationPose には2つのモードがあるが、**model-freeは事実上使用不可**。

| モード | 必要なもの | 実用性 |
|--------|-----------|--------|
| **model-based** | 3D CADメッシュ (.obj) | ✅ 実用的 |
| model-free | 参照RGBD画像 + **各フレームの既知カメラポーズ** + NeRF学習 | ❌ 極めて困難 |

model-freeは「参照画像のカメラポーズが既知」が前提。ポーズなしで使おうとするとBundleSDF(SLAMパイプライン)でポーズを先に取得する必要があり、依存関係も複雑。Isaac ROS公式もmodel-freeは非対応と明言。

### 推奨: CADメッシュを用意してmodel-basedで使う

カップのような日用品はフォトグラメトリ(スマホアプリ等)でメッシュを生成する:
- **PolyCam** (iOS/Android) - スキャンしてOBJ/GLBエクスポート
- **RealityCapture** (Windows)
- **MeshRoom** (オープンソース)

生成した `.obj` を `data/cad_models/` に置いて使用する。

### SAM3 + RealSense データ収集スクリプト

```bash
# 1. クエリ画像撮影 (.venv)
source /home/vr01/AIVisionPJ/.venv/bin/activate
python scripts/capture_realsense.py
# → data/realsense/rgb.png, depth.png, depth.npy, intrinsics.json

# 2. 参照画像撮影 (物体を手で回しながら15枚)
python scripts/capture_ref_images.py --output data/realsense/ref --n 15

# 3. SAM3でマスク生成 (.venv_sam3)
source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate
python scripts/generate_sam3_mask.py --image data/realsense/rgb.png --prompt "cup"
python scripts/generate_sam3_mask.py --dir data/realsense/ref --prompt "cup"

# 4. FoundationPose用シーンデータ整理 (.venv)
source /home/vr01/AIVisionPJ/.venv/bin/activate
python scripts/setup_fp_scene.py
# → data/fp_scene/ に rgb/, depth/, masks/, cam_K.txt, ref/ が作成される

# 5. FoundationPose実行 (Docker内, CADメッシュが必要)
docker start foundationpose_build  # 停止中の場合
docker exec -it foundationpose_build bash -c '
  source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
  export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH &&
  cd /home/vr01/AIVisionPJ &&
  python scripts/fp_create_mesh_and_run.py \
    --scene_dir data/fp_scene \
    --mesh_path data/cad_models/object.obj \
    --debug_dir results/fp_debug
'
```

### scripts/ ディレクトリ概要

| スクリプト | 用途 | 実行環境 |
|-----------|------|---------|
| `capture_realsense.py` | RealSenseで1フレーム撮影 | `.venv` |
| `capture_ref_images.py` | 参照画像セット撮影 (インタラクティブ) | `.venv` |
| `generate_sam3_mask.py` | SAM3でバイナリマスク生成 | `.venv_sam3` |
| `setup_fp_scene.py` | FoundationPose用データ整理 | `.venv` |
| `fp_create_mesh_and_run.py` | FoundationPose実行 (Poisson mesh生成も可) | Docker |
| `create_cylinder_mesh.py` | 円柱メッシュ生成 (参照用) | Docker |

### データディレクトリ構造

```
data/
├── realsense/          # RealSenseキャプチャ
│   ├── rgb.png         # クエリ画像 (BGR)
│   ├── depth.png       # 深度 (uint16 mm)
│   ├── depth.npy       # 深度 (float32 meters)
│   ├── mask.png        # SAM3マスク (binary)
│   ├── intrinsics.json # カメラ内部パラメータ
│   └── ref/            # 参照フレーム (15枚)
├── fp_scene/           # FoundationPose入力形式
│   ├── rgb/000000.png
│   ├── depth/000000.png
│   ├── masks/000000.png
│   ├── cam_K.txt
│   ├── mesh/           # 物体メッシュ
│   └── ref/            # 参照フレームコピー
└── cad_models/         # CADメッシュ置き場
```

### 将来有望な代替手法: Any6D (CVPR 2025)

FoundationPose著者(Bowen Wen)の後継研究。**単一RGBD画像1枚だけ**で6DoF推定可能。CADメッシュ不要・既知カメラポーズ不要。
- Paper: arXiv:2503.18673
- Code: https://github.com/taeyeopl/Any6D

---

## 今後の予定

- [x] SAM3: Meta承認済み・動作確認完了 (2026-03-06)
- [x] FoundationPose + RealSense: データ収集・パイプライン構築完了 (2026-03-13)
- [ ] FoundationPose model-based: CADメッシュ入手して精度検証
- [ ] Any6D: セットアップ・評価
- [ ] Qwen3-VL: 実画像でのVQAとグラウンディング
- [ ] ROS2 Humble インストールとノード実装
- [ ] YOLO26 → SAM3 → FoundationPose パイプライン統合
