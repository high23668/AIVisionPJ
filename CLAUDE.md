# AIVisionPJ

ロボット認識パイプライン: YOLO26 / DINOv3 / SAM3 / FoundationPose / Qwen3-VL / Any6D / SF3D / Depth系 を比較・統合するプロジェクト。

## 環境
- GPU: Quadro RTX 5000 Max-Q (16GB VRAM) / Ubuntu 22.04 / CUDA 13.0
- Python venv: `.venv` (uv管理) / SF3D は `.venv_3d` / SAM3 は `.venv_sam3`
- Docker: FoundationPose は `wenbowen123/foundationpose` (コンテナ名 `foundationpose_build`)
- conda 未インストール (uv 使用)

## 主要ドキュメント
- 実験レポート (透明グラス): [results/3glass_experiment_report.md](results/3glass_experiment_report.md)
- **実験レポート (ガラス板, Phase 1-7 完了): [results/glassboard_experiment_report.md](results/glassboard_experiment_report.md)**
- ガラス板実験 進捗サマリ: [docs/project_glassboard_experiment.md](docs/project_glassboard_experiment.md)
- **第2段実験 計画書: [docs/exp2/plan_exp2.md](docs/exp2/plan_exp2.md)**
- **第2段実験 ハンドオフ: [docs/exp2/handoff_exp2.md](docs/exp2/handoff_exp2.md)**

## 現在のフォーカス
**ガラス板 (148×148×2.5mm) の透明物体認識**。Phase 1〜7 完了。
- Phase 7 で FoundationPose × CAD が初成功 (7/8): Depth Pro / MoGe-2 / UniDepth V2 が板面 depth を復元
- **Depth Pro (Apple) が現時点ベスト** (境界最鮮明、Simple/Complex とも完全フィット)
- Metric3D v2 は板を透過して背景と同一視するため不適
- 確立パイプライン: `SAM3 最大面積マスク → Depth Pro → RS壁面 scale 校正 → FP × parametric CAD`

## 既知の罠
- FoundationPose `run_demo.py`: cv2.imshow が Docker でクラッシュ → `FP_SKIP_IMSHOW=1` 環境変数で完全スキップ (try/except では SIGABRT を防げない)
- FoundationPose 実行時は `QT_QPA_PLATFORM=offscreen` **不可** (offscreen プラグインが Docker container に無い)。代わりに `FP_SKIP_IMSHOW=1` を使う
- FoundationPose depth 入力は uint16 PNG (mm単位)。`.npy` の float メートル値は `* 1000` して uint16 化
- SAM3 は実行時 `DISPLAY=:0` 必須、CUDA 推論後に cv2.imshow するとセグフォルト → `xdg-open` で代替
- SF3D 入力画像は正方形前提 (横長は `crop_to_square()`)
- Flash Attention SM 7.5 (Turing) は float16 のみ、bfloat16 不可
- MoGe-2 `infer()` の `fov_x` 引数は **degrees** 単位 (radians を渡すと focal が誤推定され depth が ~63m になる)
- SAM3 `glass` プロンプトで Complex の板マスクを取る場合、最高スコアはペン瓶 (0.71) → **最大面積マスクを選ぶ**
