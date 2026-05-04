# 透明物体認識 第2段実験 再開用ハンドオフ

新規チャットでこのファイルを開いたら、まず以下を読み込んで状況把握する:

1. **`docs/exp2/plan_exp2.md`** ← 実験計画書 (ワーク仕様・シーン設計・フォルダ構成・評価指標)
2. **`results/glassboard_experiment_report.md`** ← Phase 1〜7 完了済み報告書 (背景知識)
3. **`CLAUDE.md`** ← 環境・既知の罠

---

## 1分で把握するサマリ

### Phase 1 (完了) の結論

対象: **ガラス板 148×148×2.5mm**、RealSense D435i で撮影

| 手法 | 結論 |
|------|------|
| YOLO26 | 完敗 (0 検出) |
| SAM3 | 形状◎ / 材質識別× (透明容器も拾う) |
| Qwen3-VL | 「板 vs 容器」の言語識別で唯一勝利 |
| DA2 / Marigold | 板の面を初めて depth 化、metric scale 問題あり |
| Metric3D v2 | 板を透過して壁と同一視 (不適) |
| MoGe-2 / Depth Pro / UniDepth V2 | 板面 depth 化 ✅ + RS壁面 scale 校正 → FP×CAD 成功 |
| **FoundationPose × CAD** | **Phase 7 で初成功 (7/8)。Depth Pro が現時点ベスト** |

### Phase 1 の未解決問題 → Phase 2 の動機

1. **GT なし**: RS は板を透過して壁を測定。板の真の深度が不明 → scale 校正が不正確 → CAD box が実物より小さい
2. **マスク問題**: Complex シーンで SAM3 が透明板でなくペン瓶を高スコアで選ぶ
3. **定量評価なし**: 目視比較のみで RMSE 等の数値がない

---

## Phase 2 でやること

### ワーク
- ガラス板 148×148×2.5mm (Phase 1 と同じ)
- **透明樹脂シート 70×100×0.1mm** (新ワーク)

### 撮像 (8 シーン撮影、主実験は 4 シーン)
- `glass_front` / `glass_oblique(30°)` / `resin_front` / `resin_oblique(30°)` ← **主実験**
- `{name}_cx` = distractor あり版 ← **撮影のみ、後回し**

### Ground Truth 取得 (キモ)
```
カラー紙 (赤 or 青) をワークに貼る
  → rgb_paper.png + depth_paper.npy  (RS が紙面を完璧に測定 = GT)
  → SAM3 "red paper" プロンプトで mask.png 生成
  → 紙を剥がして透明状態を撮影
```

### 評価対象: 15 モデル全部 × 4 シーン
YOLO26 / DINOv3 / SAM3 / Qwen3-VL / DA2 / Marigold / Metric3D v2 / MoGe-2 / Depth Pro / UniDepth V2 / Foundation-Stereo / ASGrasp / Boxer / Any6D / FoundationPose×CAD

---

## 現在の作業状態

| 項目 | 状態 |
|------|------|
| 計画書 | ✅ `docs/exp2/plan_exp2.md` |
| フォルダ雛形 | ⬜ 未作成 |
| CAD: glass_plate.obj | ✅ `data/glassboard_cad/glass_plate.obj` (流用) |
| CAD: resin_sheet.obj | ⬜ 未作成 (70×100×0.1mm) |
| 撮影スクリプト | ⬜ `scripts/exp2/capture_exp2.py` |
| マスク生成スクリプト | ⬜ `scripts/exp2/generate_masks_from_paper.py` |
| バッチ実行スクリプト群 | ⬜ `scripts/exp2/run_all_*.py` |
| 樹脂シート入手 | ⬜ **ユーザー側で要入手** |
| 実際の撮影 | ⬜ **樹脂シート入手後** |

---

## 次にやること (優先順)

1. **`scripts/exp2/setup_cad.py`** を実行して `data/exp2/cad/` を整備
2. **`scripts/exp2/capture_exp2.py`** を整備 (Phase 1 の `capture_realsense_dual.py` ベース)
3. **`scripts/exp2/generate_masks_from_paper.py`** を整備
4. **樹脂シート入手 → 撮影**
5. **バッチ実行スクリプト群** (`run_all_depth_models.py` など) を整備
6. **実験実行** (overnight バッチ)

---

## 環境の要点

```
プロジェクトルート: /home/vr01/AIVisionPJ/
Python venv: .venv (uv 管理)
SAM3 venv:   .venv_sam3 (Python 3.12) — 実行時 DISPLAY=:0 必須
SF3D venv:   .venv_3d (Python 3.10)
FoundationPose Docker: foundationpose_build
  起動: docker start foundationpose_build
  実行必須env: FP_SKIP_IMSHOW=1 (QT_QPA_PLATFORM=offscreen は NG)
GPU: Quadro RTX 5000 Max-Q 16GB VRAM
```

### Phase 2 で使うモデルの起動方法

```bash
# Depth Pro
.venv/bin/python scripts/exp2/run_all_depth_models.py --model depthpro

# MoGe-2 (fov_x は degrees 単位！)
.venv/bin/python scripts/exp2/run_all_depth_models.py --model moge2

# SAM3 (マスク生成)
DISPLAY=:0 .venv_sam3/bin/python scripts/exp2/generate_masks_from_paper.py

# FoundationPose
docker start foundationpose_build
bash scripts/exp2/run_all_foundationpose.sh
```

### 既存の Phase 7 スクリプト (参照・流用元)
```
scripts/phase7_depthpro.py      ← Depth Pro 推論パターン
scripts/phase7_moge2.py         ← MoGe-2 推論パターン (fov_x degrees 注意)
scripts/phase7_metric3d.py      ← Metric3D 推論パターン
scripts/phase7_unidepth.py      ← UniDepth 推論パターン
scripts/phase7_run_foundationpose.sh ← FP 実行パターン
```

---

## Phase 1 振り返りから来る改善点

### 第2段実験で実施する (スクリプトに組み込む)
- **複数フレーム平均**: RS depth を 10〜30 フレーム平均 → `capture_exp2.py`
- **MoGe-2 法線保存**: `normal_moge2.npy` + 板面角度誤差を metrics に追加 → `run_all_depth_models.py`
- **GT校正 vs 壁ベース校正の比較記録**: 両方の Z 誤差を並記 → `evaluate_depth_gt.py`
- **SAM3 マスク IoU**: 透明状態 SAM3 マスク vs 紙GT マスクの IoU → `evaluate_depth_gt.py`
- **FP score 分布幅の記録**: 252 候補のスコア range → `fp_pose_errors.csv`
- **樹脂シート FP 用 CAD 厚み**: 0.1mm → FP 用は 1mm に設定 → `setup_cad.py`

### 今後の課題 (Phase 3 以降)
- UniDepth confidence を FP ICP に反映 (FP 本体改造が必要)
- 本番運用 (GT なし) での最適校正方法を第2段結果から決定
- 30° 斜め結果から透明物体の反射角度依存を考察
- CAD box サイズ問題の根本原因 (GT で確定)
- FP score 分離改善しない場合は RGB texture 併用モード検討

→ 詳細は `docs/exp2/plan_exp2.md` §12

---

## フォルダ構成 (完成形)

```
data/exp2/
├── cad/
│   ├── glass_plate.obj    148×148×2.5mm
│   └── resin_sheet.obj    70×100×0.1mm
├── glass_front/
│   ├── rgb.png, rgb_paper.png
│   ├── depth.npy, depth_paper.npy (GT)
│   ├── left_ir.png, right_ir.png
│   ├── intrinsics.json
│   ├── mask.png          (紙GT画像 → SAM3 生成)
│   └── measurements.json (実測値 + gt_z_center_mm)
├── glass_oblique/   (同構造)
├── resin_front/     (同構造)
├── resin_oblique/   (同構造)
└── {}_cx/           (distractor版 4シーン、後回し)

results/exp2/
├── depth/            各モデルのdepth可視化 + GT誤差マップ
├── rgb_models/       YOLO/SAM3/DINO/Qwen結果
├── foundationpose/   FP×CAD可視化
├── metrics/          depth_errors.csv, fp_pose_errors.csv
└── exp2_report.md

scripts/exp2/
├── setup_cad.py
├── capture_exp2.py
├── generate_masks_from_paper.py
├── compute_gt_z.py
├── run_all_depth_models.py
├── run_all_rgb_models.py
├── evaluate_depth_gt.py
├── setup_fp_scenes.py
├── run_all_foundationpose.sh
└── compile_report.py
```

---

## 評価指標

| 指標 | 対象 | 意味 |
|------|------|------|
| Plate RMSE (m) | depth モデル | 板面の深度精度 |
| Inlier@5cm / @2cm | depth モデル | 実用閾値内の画素割合 |
| FP Z error (mm) | FoundationPose | 奥行き位置精度 |
| FP 姿勢誤差 (°) | FoundationPose | 回転精度 |
| SAM3 IoU | セグメンテーション | GT マスクとの一致率 |

---

## measurements.json の構造

```json
{
  "scene": "glass_front",
  "camera_to_object_front_mm": 300,
  "object_thickness_mm": 2.5,
  "object_to_wall_mm": 120,
  "object_tilt_deg": 0,
  "paper_color": "red",
  "gt_z_center_mm": 312.4,
  "notes": ""
}
```

`gt_z_center_mm` は `compute_gt_z.py` が `depth_paper.npy` から自動計算して書き込む。

---

## Scale 校正方針 (Phase 2 改定版)

```python
# GT が使えるので 2 段階校正が正確にできる
s_wall  = median(RS_wall) / median(model_wall)     # 壁基準 (従来)
depth_wall_cal = depth_raw * s_wall

alpha   = gt_z_center / median(depth_wall_cal[mask])  # GT基準の補正
depth_final = depth_wall_cal * alpha
```

---

## 参考: Phase 1 フォルダとの違い

| 項目 | Phase 1 | Phase 2 |
|------|---------|---------|
| フォルダ | `data/glassboard_simple/` など散在 | `data/exp2/{scene}/` に集約 |
| GT | なし (RS は壁を測定) | `depth_paper.npy` が真の板面 depth |
| マスク | SAM3 で誤認識あり | カラー紙画像から SAM3 → 高精度 |
| Scale 校正 | 壁基準のみ → Z 過大 | GT Z で補正 → 正確 |
| 評価 | 目視のみ | RMSE / MAE / Inlier rate 定量評価 |
