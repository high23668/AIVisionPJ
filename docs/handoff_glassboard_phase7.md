# ガラス板実験 Phase 7 再開用ハンドオフ

新規チャットでこのファイルを開いたら、まず以下の2点を読み込んで状況把握する (パスはプロジェクトルート `/home/vr01/AIVisionPJ/` 起点):

1. **`results/glassboard_experiment_report.md`** ← 実験本体 (Phase 1〜6完了, §9 が Phase 6)
2. **`docs/project_glassboard_experiment.md`** ← 進捗・既知結論サマリ

## ここまでのまとめ (1分で把握)

対象: **ガラス板 148×148×厚2.5mm**、RealSense D435i で撮影 (simple/complex 2シーン)

| Phase | 手法 | 結論 |
|-------|------|------|
| 1 | YOLO26 / DINOv3 / SAM3 / Qwen3-VL | Qwen3-VL が「板vs容器」の言語識別で MVP。SAM3 は形状◎だが材質×、YOLO26 は完敗 |
| 2 | Boxer / Any6D × RS depth | 厚み17cm過大、Any6D Simple は板にフィット |
| 3 | Foundation-Stereo | 板の "面" は不可視、エッジのみ |
| 4 | Boxer / Any6D × FS depth | RS版とほぼ同等 |
| 5 | ASGrasp | 板を grasp 対象としない (DREDS学習バイアス + 物理的に2.5mm掴めない) |
| 6 | DA2 / Marigold / FP×CAD | DA2 と Marigold で**板の面を初めて密に depth 化** (大きな前進)。ただし DA2 は metric が ~5x 過大、Marigold は相対値のみ。**FP×CAD は 4/4 失敗** (depth ICP が壁/床/容器に引っ張られる) |

ReFlow6D は pretrained 未公開のため断念済み。

## Phase 7 でやりたいこと

**透明物体に効きそうな最新の単眼 depth モデル** をさらに評価し、scale 問題を解消できるか検証:

| モデル | 種別 | 期待される効果 |
|--------|------|---------------|
| **Depth Pro** (Apple) | metric | 境界精度最高クラス、ガラス板の薄エッジ捕捉に期待 |
| **UniDepth V2** | metric | intrinsic 不要、不確実性出力あり (ガラス領域の信頼度が下がるか確認) |
| **MoGe-2** | metric | 点群・法線・depth 一括取得、3D OBB 直接構築の可能性 |
| **Metric3D v2** | metric | 法線同時推定、SLAM 統合実績 |

**評価軸** (Phase 6 と同じ):
- 形状: 板の面が depth に出るか
- Metric 寸法: 実距離 ~30cm に対する誤差 (DA2 は ~5x 過大が baseline)
- ハイブリッド: その depth を FoundationPose × CAD に流して 6DoF が解けるか

## 入力データ (既存、再撮影不要)

```
data/glassboard_{simple,complex}/
├── rgb.png              入力RGB
├── depth.npy            RS depth (校正用 baseline)
├── intrinsics.json
├── mask.png             SAM3 生成マスク
├── depth_da2.npy        Phase 6 DA2 出力 (比較用)
└── depth_marigold.npy   Phase 6 Marigold 出力 (比較用)

data/glassboard_cad/
└── glass_plate.obj      148×148×2.5mm parametric box CAD
```

## Phase 7 の進め方テンプレ

各モデル個別に以下を実施 (Phase 6 で確立したパターン):

1. モデル単体で simple/complex 2シーンの depth 推定
2. 可視化 (`results/glassboard/<model>_{simple,complex}.jpg`)
3. RS depth の壁面領域で metric scale を校正 (DA2/Marigold で必要だったやり方)
4. 校正後 depth を `data/glassboard_{simple,complex}_fp_<model>/depth/000000.png` (uint16 mm) として保存
5. FoundationPose × parametric CAD で 6DoF 推定
6. `results/glassboard/foundationpose_<model>_{simple,complex}.jpg` で可視化
7. 報告書 `results/glassboard_experiment_report.md` に Phase 7.x として追記

## 環境の要点 (再掲)

- Python venv: `/home/vr01/AIVisionPJ/.venv` (uv管理)
- SF3D 用: `.venv_3d` (Python 3.10)
- SAM3 用: `.venv_sam3` (Python 3.12, 実行時 `DISPLAY=:0` 必須)
- FoundationPose Docker: コンテナ名 `foundationpose_build`、`run_demo.py` を `-v` でマウント
- FoundationPose 実行時: `QT_QPA_PLATFORM=offscreen` 必須 (cv2.imshow 失敗回避)
- GPU: Quadro RTX 5000 Max-Q (16GB VRAM) — Depth Pro 等の重いモデルは VRAM に注意

## 既知の罠 (Phase 6 で踏んだもの)

- **FoundationPose `run_demo.py` の cv2.imshow** はホスト Docker でクラッシュ → `try/except` で囲み、`imwrite` を常に走らせるよう既に改修済み (この変更は維持すること)
- **DA2/Marigold 可視化の vmin/vmax は per-source auto-scale** (RS の固定範囲を使い回すと真っ赤になる)
- **FP × CAD は depth に板の面が無いと解けない** ─ Phase 7 でも depth 校正後に「板領域に正しい距離値があるか」を必ず可視化で確認する
- **FoundationPose の depth 入力は uint16 PNG (mm単位)** ─ `.npy` の float メートル値を `* 1000` して uint16 化が必要

## 期待する Phase 7 の最終アウトプット

- `results/glassboard_experiment_report.md` に §10 (Phase 7) を追加
- 4 モデル × 2 シーン × (depth単体 + FP統合) = 約 16 枚の図
- 全体サマリ表に 4 モデル行を追加
- 結論: 「DA2/Marigold/Depth Pro/UniDepth/MoGe-2/Metric3D2 のうち、ガラス板で metric も形状も両立する勝者はどれか」
