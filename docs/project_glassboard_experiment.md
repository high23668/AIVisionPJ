# ガラス板透明物体認識実験 ─ 進捗サマリ (Phase 1-6完了 / Phase 7予定)

148×148×2.5mm ガラス板に対する全モデル比較実験の進捗・既知結論・次フェーズ単眼depth候補。

## 対象
ガラス板 148 × 148 × 厚2.5mm をRealSense D435i で撮影 (simple/complex 2シーン)
データ: `data/glassboard_{simple,complex}/`、CAD: `data/glassboard_cad/glass_plate.obj`
報告書: `results/glassboard_experiment_report.md` (Phase 1〜6 完了)

## 完了フェーズと結論
- Phase 1 (RGB系): YOLO26 完敗、SAM3 形状◎・材質識別×、Qwen3-VL がディストラクタ識別 MVP
- Phase 2 (RS depth × Boxer/Any6D): 厚み17cm過大、Any6D Simple は板にフィット
- Phase 3 (Foundation-Stereo): 板の "面" は不可視、エッジのみ復元
- Phase 4 (FS depth × Boxer/Any6D): RS版とほぼ同等、根本限界変わらず
- Phase 5 (ASGrasp): 板を grasp 対象とせず (DREDS学習バイアス + 厚2.5mmで物理的に掴めない)
- Phase 6 (DA2 / Marigold / FP×CAD):
  - DA2 Indoor Metric: 板の面を初めて密に復元、metric値は~5x過大
  - Marigold v1.1: 形状品質はDA2同等以上、相対値のみ
  - FoundationPose × parametric CAD: 4/4失敗 (depth ICPが壁・床・容器に引っ張られる)
- ReFlow6D は pretrained 未公開で断念

**Why:** ガラス板は容器より深刻に難しい。RGB特徴希薄＋IR/depth透過＋把持困難
**How to apply:** 同種実験を再開する場合、「板の面が観測depthに無いとCADがあっても解けない」を前提に手法選定する

## Phase 7 完了 (2026-05-02〜03)

### 評価結果サマリ

| モデル | 板の面 depth | metric | 境界 | FP×CAD成功 |
|--------|------------|--------|------|-----------|
| Metric3D v2 ViT-L | ✗ 壁に溶込み | ◯ | △ | 1/2 |
| MoGe-2 ViT-L+normal | ✅ 壁より前 | ◯ | ◯ | 2/2 |
| **Depth Pro** | ✅ **最鮮明** | ◯ | ✅ | **2/2** |
| UniDepth V2 ViT-L | ◯ 弱め | ◯+conf | △ | 2/2 |

**FP × CAD: 7/8 成功** (Phase 6 は 0/4)

**Why:** Phase 6 で板面 depth 化が確認できたので metric + 高精度モデルで scale 解消 → FP 成功
**How to apply:** 「Depth Pro または MoGe-2 で板面 depth → RS壁面 scale 校正 → FP × parametric CAD」が実証済みパイプライン

### 確立したパイプライン
```
SAM3 最大面積マスク (= 板) →
Depth Pro / MoGe-2 → depth_{model}.npy →
scale = median(RS_wall) / median(model_wall) →
uint16 PNG (mm) → FoundationPose × glass_plate.obj
```

### 重要な知見
- Metric3D v2 は透明板を「背景と同一」と扱うため板面を捕捉できない
- Complex の板マスクは SAM3 `glass` プロンプトの「最大面積」で選ぶ (最高スコアだとペン瓶になる)
- MoGe-2 の `fov_x` 引数は degrees 単位 (radians を渡すと depth が ~63m オフセットになる)
- FP の cv2.imshow は offscreen Qt プラグインが無いと SIGABRT → `FP_SKIP_IMSHOW=1` 環境変数で完全スキップ
