# 透明物体認識 第2段実験 計画書

**作成日**: 2026-05-03  
**対象**: ガラス板 / 樹脂シート × 正面・斜め撮影  
**目的**: Phase 1 の課題 (GT なし / マスク問題 / scale 校正不正確) を解消した上で全モデルを再評価し、定量的な比較を行う

---

## 1. 背景と Phase 1 からの改善点

| 課題 | Phase 1 状態 | Phase 2 対策 |
|------|------------|-------------|
| Ground Truth がない | RS は透明物体を透過して壁を測定 → 板距離 unknown | **カラー紙をワークに貼って RS で板面 GT を取得** |
| Scale 校正が不正確 | 壁基準の単一スカラー校正 → 板 Z が ~40% 過大 | GT Z で直接校正 |
| Complex マスク問題 | SAM3 が透明物体より手前のペン瓶を高スコアで選ぶ | **カラー紙 (赤 or 青) を貼った状態で SAM3 → 完全自動・高精度** |
| 定量評価なし | 目視のみ | GT 比較: RMSE / MAE / Inlier rate / FP Z 誤差 |
| フォルダが散乱 | Phase 1〜7 が `data/glassboard_*` に雑多 | `data/exp2/` 以下に完全集約 |

---

## 2. 評価対象モデル (全 15 モデル)

### 2.1 RGB 系 (検出・セグメンテーション・理解)
| # | モデル | 評価内容 |
|---|--------|---------|
| 1 | YOLO26 (YOLOv8-L) | 2D bbox 検出率 |
| 2 | DINOv3 | PCA 可視化・類似度マップ |
| 3 | SAM3 | マスク精度 (GT マスクと IoU) |
| 4 | Qwen3-VL 4B | `glass plate` / `transparent sheet` 言語識別精度 |

### 2.2 Depth 系 (単眼深度推定)
| # | モデル | 種別 | 備考 |
|---|--------|------|------|
| 5 | Depth Anything V2 Indoor Metric | metric | Phase 6 で板面初復元 |
| 6 | Marigold v1.1 | 相対 | Phase 6、高品質境界 |
| 7 | Metric3D v2 ViT-Large | metric | Phase 7、ガラスを透過する傾向 |
| 8 | MoGe-2 ViT-L+normal | affine+FoV | Phase 7、法線・点群も取得 |
| 9 | Depth Pro (Apple) | metric | Phase 7 現時点ベスト |
| 10 | UniDepth V2 ViT-L | metric+conf | Phase 7、confidence map が特徴 |

### 2.3 Stereo / Grasp
| # | モデル | 評価内容 |
|---|--------|---------|
| 11 | Foundation-Stereo | ステレオ depth、エッジ検出 |
| 12 | ASGrasp | grasp スコア・把持候補位置 |

### 2.4 3D ポーズ推定
| # | モデル | 評価内容 |
|---|--------|---------|
| 13 | Boxer | 3D OBB サイズ・位置精度 |
| 14 | Any6D (SF3D) | model-free 6DoF ポーズ誤差 |
| 15 | FoundationPose × parametric CAD | 6DoF ポーズ誤差 (Z 誤差 mm / 姿勢誤差 °) |

---

## 3. ワーク仕様

| ワーク | サイズ (mm) | 厚み | CAD ファイル |
|--------|-----------|------|------------|
| ガラス板 | 148 × 148 | 2.5mm | `data/exp2/cad/glass_plate.obj` (コピー) |
| 透明樹脂シート | 70 × 100 | **0.1mm** | `data/exp2/cad/resin_sheet.obj` (新規) |

樹脂シートは PET または薄アクリルを想定。0.1mm は FP の ICP では厚みは事実上無視され、板面検出が主目的。

---

## 4. 撮像シーン設計

### 4.1 主実験シーン (4 シーン)

| シーン名 | ワーク | 角度 | distractor |
|---------|--------|------|-----------|
| `glass_front` | ガラス板 | 正面 (0°) | なし |
| `glass_oblique` | ガラス板 | **30° 傾き** | なし |
| `resin_front` | 樹脂シート | 正面 | なし |
| `resin_oblique` | 樹脂シート | 30° | なし |

### 4.2 撮影のみシーン (後回し、4 シーン)

| シーン名 | 内容 |
|---------|------|
| `glass_front_cx` | ガラス板正面 + distractor (容器等) |
| `glass_oblique_cx` | ガラス板30° + distractor |
| `resin_front_cx` | 樹脂シート正面 + distractor |
| `resin_oblique_cx` | 樹脂シート30° + distractor |

distractor シーンは同条件で撮影しておき、モデル実行は後続フェーズへ。

### 4.3 斜め (30°) 撮影の留意点

- ワーク表面の法線がカメラ光軸と 30° ずれる
- depth_paper.npy の中央 median = `gt_z_center` (GT 奥行き)
- 板の端では depth が ±L/2 × sin(30°) = ±25mm (ガラス板) / ±17.5mm (樹脂) 分だけ前後にずれる
- FP の ICP がこの傾き depth を正しく処理できるかがテストポイント

---

## 5. 撮影手順 (1 シーンあたり)

```
Step 1: 三脚固定・シーンセットアップ
Step 2: メジャー計測 → measurements.json 記録
          camera_to_object_front_mm: カメラ前面〜ワーク前面
          object_thickness_mm:       ワーク厚み
          object_to_wall_mm:         ワーク後面〜壁
          object_tilt_deg:           傾き角 (0 or 30)
          paper_color:               "red" or "blue"

Step 3: カラー紙 (赤 or 青) をワーク全面に貼る
          → 撮影: rgb_paper.png, depth_paper.npy
          → scripts/exp2/generate_masks_from_paper.py で mask.png 生成
          → scripts/exp2/compute_gt_z.py で gt_z_center_mm を measurements.json に追記

Step 4: 紙を剥がして透明状態に戻す
          → emitter ON:  rgb.png, depth.npy
          → emitter OFF: left_ir.png, right_ir.png
```

---

## 6. Ground Truth の定義

### 6.1 Depth GT
```
GT depth map = depth_paper.npy  (RS が紙面を直接測定)
gt_z_center  = median(depth_paper.npy[mask])  [m]
```

### 6.2 FP Pose GT
```
gt_translation = [x, y, gt_z_center]  (x, y は FP 推定値を使用、Z のみ GT で検証)
gt_rotation    = 計測角度から計算した理論姿勢行列
```

---

## 7. マスク生成方針

```
rgb_paper.png (カラー紙)
  ↓ SAM3 プロンプト: "red paper" or "blue paper"
  ↓ 最高スコア or 最大面積 (紙は不透明なので高スコア確実)
  → mask.png (透明ワーク評価に流用)
```

Phase 1 の問題 (透明→低スコア、ペン瓶と混同) を構造的に解消。

---

## 8. Scale 校正方針 (Phase 2 改定版)

Phase 1 の問題: 壁基準の単一スカラー → 板 Z が ~40% 過大

Phase 2 の校正:
```
alpha = gt_z_center / model_plate_depth_after_wall_calibration
depth_final = depth_wall_calibrated * alpha
```

GT があるため alpha が正確に計算できる。FP scene 生成前に自動適用。

---

## 9. 評価指標

| 指標 | 計算方法 | 単位 |
|------|---------|------|
| Plate RMSE | `sqrt(mean((pred-gt)^2))` on mask | m |
| Plate MAE | `mean(|pred-gt|)` on mask | m |
| Inlier@5cm | mask 内で `|pred-gt| < 0.05` の割合 | % |
| Inlier@2cm | mask 内で `|pred-gt| < 0.02` の割合 | % |
| FP Z error | `|pose_Z - gt_z_center|` | mm |
| FP 姿勢誤差 | rotation matrix → geodesic distance | ° |

全指標を `results/exp2/metrics/depth_errors.csv` に出力し、モデル横断比較表を生成。

---

## 10. フォルダ構成

```
data/exp2/
├── cad/
│   ├── glass_plate.obj          148×148×2.5mm
│   └── resin_sheet.obj          70×100×0.1mm
├── glass_front/
│   ├── rgb.png                  透明状態 RGB (評価入力)
│   ├── rgb_paper.png            カラー紙貼り付け RGB (GT 用)
│   ├── depth.npy                RS depth (透明)
│   ├── depth_paper.npy          RS depth (紙付き = GT)
│   ├── left_ir.png              emitter OFF 左 IR
│   ├── right_ir.png             emitter OFF 右 IR
│   ├── intrinsics.json
│   ├── mask.png                 紙 GT 画像から SAM3 で生成
│   └── measurements.json
├── glass_oblique/               (同構造)
├── resin_front/                 (同構造)
├── resin_oblique/               (同構造)
├── glass_front_cx/              (撮影のみ、後続フェーズ)
├── glass_oblique_cx/
├── resin_front_cx/
└── resin_oblique_cx/

results/exp2/
├── depth/
│   ├── {model}_{scene}.jpg
│   └── {model}_{scene}_error.jpg
├── rgb_models/
│   ├── yolo26_{scene}.jpg
│   ├── sam3_{scene}.jpg
│   ├── dinov3_{scene}_{pca,sim}.jpg
│   └── qwen3vl_{scene}.jpg
├── foundationpose/
│   └── fp_{model}_{scene}.jpg
├── metrics/
│   ├── depth_errors.csv
│   └── fp_pose_errors.csv
└── exp2_report.md

scripts/exp2/
├── setup_cad.py                 CAD 生成 (resin_sheet.obj)
├── capture_exp2.py              撮影スクリプト
├── generate_masks_from_paper.py rgb_paper.png → SAM3 → mask.png
├── compute_gt_z.py              GT Z 計算 → measurements.json 更新
├── run_all_depth_models.py      全 depth モデル × 全シーン 一括
├── run_all_rgb_models.py        RGB 系モデル 一括
├── evaluate_depth_gt.py         GT 比較・CSV 出力
├── setup_fp_scenes.py           FP scene dir 一括生成
├── run_all_foundationpose.sh    FP 全ケース一括
└── compile_report.py            結果集約 → exp2_report.md 生成
```

---

## 11. 実施スケジュール

| ステップ | 内容 | 目安時間 |
|---------|------|---------|
| 準備 | 樹脂シート入手・CAD 作成・スクリプト整備 | 1〜2時間 (PC作業) |
| 撮影 | 8 シーン × 4 ショット = 32 ショット | 1〜2時間 (実機) |
| 前処理 | マスク生成・GT Z 計算・FP scene 構築 | 30分 (自動) |
| 実験実行 | 全モデル × 4 シーン (バッチ自動) | ~1日 (overnight) |
| レポート | 定量比較表・可視化 | 半日 |

---

## 12. Phase 1 振り返りから得た改善事項

### 12.1 第2段実験で実施する改善 (スクリプトに組み込む)

| # | 改善内容 | 組み込み先 |
|---|---------|-----------|
| ① | **複数フレーム平均**: RS depth を 10〜30 フレーム平均して壁面校正 anchor を安定化 | `capture_exp2.py` |
| ② | **MoGe-2 法線を保存・可視化**: `normal_moge2.npy` として保存し、板面法線の角度誤差を評価指標に追加 | `run_all_depth_models.py` |
| ③ | **GT校正 vs 壁ベース校正の比較記録**: Phase 2 の GT Z を使った校正と、GT なしの壁ベース校正の誤差を並記。本番運用時の校正精度見積もりに使う | `evaluate_depth_gt.py` |
| ④ | **SAM3 マスク IoU 評価**: 透明状態の SAM3 マスクと紙GT マスクの IoU を計測。透明物体 segmentation 精度の定量指標 | `evaluate_depth_gt.py` |
| ⑤ | **FP score 分布幅の記録**: 252 候補ポーズのスコア range (max-min) を収録。depth 品質の proxy 指標として使用 | `run_all_foundationpose.sh` → `fp_pose_errors.csv` に追記 |
| ⑥ | **樹脂シート CAD の厚み調整**: 0.1mm は FP ICP でゼロ扱いになりうるため、FP 用 CAD は 1mm に設定して安定化を図る (寸法測定とは別管理) | `setup_cad.py` |

### 12.2 今後の課題 (第2段以降で検討)

| # | 課題 | 内容 |
|---|------|------|
| A | **UniDepth confidence を FP ICP に反映** | FP 本体の深度重み付けを改造して板領域低 confidence を down-weight。工数大きいため Phase 3 以降 |
| B | **本番運用時の GT なし校正戦略** | ロボット実機では GT (カラー紙) が使えない。第2段の結果で「壁ベース / スタンドベース / affine」のどれが GT に最も近いかを確認し、Phase 3 の校正方針を決める |
| C | **30° 斜め × 透明物体の光学モデル** | 反射率の角度依存 (Brewster 角 ~56°) により depth モデルが板面を見えやすくなる可能性を検証。第2段の oblique 結果から考察 |
| D | **Metric3D v2 が透過する原因の解明** | 学習データ (outdoor SLAM 主体) vs Depth Pro (mixed indoor) の違いを文献確認し報告書考察を補強 |
| E | **CAD box サイズ問題の根本解明** | Phase 1 では ratio 校正後も ~15% 小さかった。板傾き・マスク過剰セグメンテーション・スタンド誤差のどれが原因か第2段 GT で確定 |
| F | **FP ICP の収束安定性向上** | Phase 1 でスコア range が狭く (112〜115)、どの pose も「同程度に良い」状態だった。depth 品質向上 (GT 校正) でスコア分離が改善するか確認。改善しなければ RGB texture も使う FP モード (model-free) との組み合わせを検討 |

---

## 13. 既知の落とし穴 (Phase 1 から継承)

- FoundationPose `run_demo.py`: `FP_SKIP_IMSHOW=1` 必須 (`QT_QPA_PLATFORM=offscreen` は NG)
- MoGe-2 `infer()` の `fov_x` は **degrees** 単位 (radians を渡すと depth ~63m)
- SAM3 実行時は `DISPLAY=:0` 必須、CUDA 後の `cv2.imshow` はセグフォルト → `xdg-open` 代替
- FoundationPose depth 入力は uint16 PNG (mm 単位)。float meters × 1000 して uint16 化
- Flash Attention SM 7.5 は float16 のみ (bfloat16 不可)
- Depth Pro の checkpoint は `models/depth_pro/checkpoints/depth_pro.pt` にある (HF キャッシュではない)
