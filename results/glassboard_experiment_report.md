# ガラス板 透明物体認識 比較実験報告書

**実験日**: 2026-04-26
**実施者**: high2366834@gmail.com
**実験環境**:
- OS: Ubuntu 22.04 / Linux 6.8.0
- GPU: Quadro RTX 5000 Max-Q (16GB VRAM)
- センサー: RealSense D435i
- 対象物体: **ガラス板 148 × 148 × 厚2.5 mm**

---

## 1. 本実験の位置づけ

[3glass実験](3glass_experiment_report.md) は**透明な"容器"** (グラス) を対象にしたが、本実験は**透明な"板"** (薄板) を対象とする。

容器との重要な差異：

| 観点 | 透明グラス（容器） | 透明ガラス板（薄板） |
|------|-----------------|------------------|
| 形状 | 円筒状の体積物体 | **平面・厚み2.5mm の極薄板** |
| RGB上の見え方 | 縁＋反射＋歪みで存在感あり | **背景がほぼ素通し** |
| IR上の見え方 | 縁が見える | **ほぼ透過、エッジ部のみ** |
| RealSense IR depth | 透明部分は欠損 | **板を透過して背景の壁面depthが取れてしまう** |
| 3D OBB | 円筒形 OBB を期待 | **板状（厚み2.5mm）の極薄OBBを期待** |

→ **薄板は容器より深刻に難しい**。RGBで見えにくく、深度センサーは「見えない」と認識する。

実験対象は ReFlow6D 後段試行の前提で、**CADを後で簡単にパラメトリック生成**できる単純形状（直方体）にした。

---

## 2. 撮影プラン

`scripts/capture_realsense_dual.py` でデュアルモード撮影：
- **emitter ON** で RGB + RS depth
- **emitter OFF** で 左右IR（Foundation-Stereo / ASGrasp 用）

両モード <1 秒で同一シーンを保証。

### 2.1 Simple シーン

ガラス板 1枚を木目床に立てて撮影。背景は壁のみ。

![撮影プレビュー Simple](glassboard/capture_preview_simple.jpg)
*キャプション: Simple シーン。RGB / 左IR / 右IR の同一シーン取得結果。RGBではガラス板が**ほぼ透明にしか映らず**、輪郭と中央の鏡反射、底のスタンドのみ確認できる。*

![入力 RGB Simple](glassboard/input_simple.png)
*キャプション: Phase 1 ~ 5 すべてで使用する Simple シーンの RGB 入力。*

### 2.2 Complex シーン

ガラス板の手前にペン瓶（ガラス容器）とスプレーボトル（プラ容器）を配置。**ガラス板を通して両容器が透けて見える**構図。

![撮影プレビュー Complex](glassboard/capture_preview_complex.jpg)
*キャプション: Complex シーン。RGB / 左IR / 右IR。手前左にペン瓶、手前右にスプレーボトル、奥にガラス板。両容器の上半分はガラス板を通して見える状態。*

![入力 RGB Complex](glassboard/input_complex.png)
*キャプション: Phase 1 ~ 5 すべてで使用する Complex シーン RGB。*

### 2.3 撮影統計

| シーン | RealSense IR depth 欠損率 | コメント |
|-------|----------------------|---------|
| Simple | **5.7%** | **板を透過して壁面depthが取れる** ためほぼ全画素有効 |
| Complex | 16.1% | ペン瓶・スプレー部分の欠損 + 板でなく容器側の欠損が支配的 |

→ ガラス板は **RealSenseセンサーから見ると "見えていない" のと等価**。
**「板そのもの」のdepthは取れず、その奥にある壁・床が記録されている**。これがガラス板特有の難しさ。

---

## 3. Phase 1 ─ RGB系4モデル評価

### 3.1 YOLO26（2D物体検出）

**コマンド**:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo26l.pt')(rgb_path, conf=0.15)"
```

**結果**:

| シーン | 検出 | コメント |
|-------|------|---------|
| Simple | **0個** | ガラス板を全く検出せず |
| Complex | bottle 0.93 (スプレーボトル のみ) | 板・ペン瓶も検出せず |

![YOLO26 Simple](glassboard/yolo26_simple.jpg)
*キャプション: YOLO26 (Simple)。**何も検出されない**。COCO 80クラスに「flat glass / mirror / pane」相当のラベルがなく、また形状的に「板」は特徴を捕捉しにくい。*

![YOLO26 Complex](glassboard/yolo26_complex.jpg)
*キャプション: YOLO26 (Complex)。スプレーボトルのみ bottle 0.93 で検出。**ガラス板 + ペン瓶 は完全に見逃し**。*

**所見**:
- **YOLO26 はガラス板に対して完敗**（Simple 0検出）
- COCO クラス制約 + 透明薄板の特徴量希薄性が原因
- ガラス板検出には別系統（OWLv2 / DINOv3 / VLM）が必要

---

### 3.2 DINOv3（視覚特徴抽出）

PCA可視化と類似度マップ（中央のガラス板上にアンカー）を生成。

#### PCA 可視化

![DINOv3 PCA Simple](glassboard/dinov3_simple_pca.jpg)
*キャプション: DINOv3 PCA (Simple)。板領域は背景と若干異なる色（鏡反射部分は別クラスタ）として分離されるが、**板全体は背景にほぼ埋没**。*

![DINOv3 PCA Complex](glassboard/dinov3_complex_pca.jpg)
*キャプション: DINOv3 PCA (Complex)。前列のペン瓶・スプレーボトルが明確なクラスタとして浮き上がるが、**ガラス板自体は背景に近い色**（透過していて固有特徴がほぼないため）。*

#### 類似度マップ

![DINOv3 類似度 Simple](glassboard/dinov3_simple_sim.jpg)
*キャプション: DINOv3 類似度 (Simple, アンカー=板中央)。**鏡反射の局所領域**だけが赤く浮かぶ。板全体（画面の大半）は捕捉できず、**「ガラス板」ではなく「映り込みの模様」**を特徴として捉えている。*

![DINOv3 類似度 Complex](glassboard/dinov3_complex_sim.jpg)
*キャプション: DINOv3 類似度 (Complex, アンカー=板上部)。板中央域が赤くハイライトされ、ペン瓶・スプレーは青（低類似）。**ある程度識別可能**。*

**所見**:
- DINOv3 は「ガラス板」自体を強く捉える特徴を持たない（背景透過のため）
- 鏡反射の領域、および容器を背景にした部分でかすかに反応
- Complex のほうが Simple より検出しやすい（前景物体との対比で）

---

### 3.3 SAM3（テキストプロンプトセグメンテーション）

複数のプロンプトで実験。

| シーン | プロンプト | マスク数 | スコア | 結果 |
|-------|----------|---------|--------|------|
| Simple | `glass plate` | 0 | — | 検出なし |
| Simple | `glass` | **1** | **0.84** | ✅ 板全体を正確にマスク |
| Simple | `transparent plate` | 0 | — | 検出なし |
| Simple | `glass board` | 1 | 0.63 | ✅ 板捕捉、スコア低め |
| Complex | `glass plate` | 0 | — | 検出なし |
| Complex | `glass` | **3** | 0.52, 0.71, 0.65 | ✅ 板 + ペン瓶 + スプレーボトル |
| Complex | `transparent plate` | 0 | — | 検出なし |
| Complex | `glass board` | 0 | — | 検出なし |

![SAM3 Simple `glass`](glassboard/sam3_simple_glass.jpg)
*キャプション: SAM3 (Simple, prompt=`glass`)。**ガラス板全体をスタンドまで含めて完璧に緑マスク化**。スコア 0.84。本実験で最も精度の高い形状認識。*

![SAM3 Complex `glass`](glassboard/sam3_complex_glass.jpg)
*キャプション: SAM3 (Complex, prompt=`glass`)。3 個マスク：紫=ガラス板 (0.65)、オレンジ=ペン瓶 (0.71)、緑=スプレーボトル (0.52)。**「glass」プロンプトでは透明物体全般を区別せず拾う**特性。*

**所見**:
- SAM3 は `glass` プロンプトで**ガラス板全体を高スコア (0.84) で完璧マスク化**できる
- Simple では本実験でも最強の形状認識
- Complex では透明な容器（ペン瓶・スプレー）も同列に拾う → **材質識別はできない**
- `glass plate` `transparent plate` `glass board` プロンプトはスコア 0 で全く効かない（学習データにないため）

---

### 3.4 Qwen3-VL 4B（VLM）

複数のプロンプトで grounding を実行。

| シーン | プロンプト | 検出bbox | コメント |
|-------|----------|---------|---------|
| Simple | glass plate | **1** ✅ | 板全体の正確な bbox |
| Simple | transparent glass board | **1** ✅ | 同様に正確 |
| Simple | glass | **1** ✅ | 同様 |
| Complex | glass plate | **1** ✅ | 板の底面エッジ部分の bbox（板全体は透明過ぎて意味的エッジを優先） |
| Complex | transparent glass board | **1** ✅ | 同様 |
| Complex | glass | 2 | bottle×2 を返す（ペン瓶+スプレー、**板でなく "ガラス容器" を解釈**） |

![Qwen3-VL Simple `glass plate`](glassboard/qwen3vl_simple_glass_plate.jpg)
*キャプション: Qwen3-VL grounding (Simple, prompt=`glass plate`)。板全体の領域を bbox で正確に囲む。*

![Qwen3-VL Complex `glass plate`](glassboard/qwen3vl_complex_glass_plate.jpg)
*キャプション: Qwen3-VL grounding (Complex, prompt=`glass plate`)。板の底面エッジ部分（スタンド+底辺ライン）を bbox 化。**「板」の存在を概念的に拾えている**ことの証左。*

**所見**:
- Qwen3-VL は **言語的概念で「glass plate」と「glass containers」を区別**できる本実験唯一のモデル
- `glass plate` プロンプトではガラス板を、`glass` プロンプトではガラス容器を返す
- bbox 精度は SAM3 より粗いが、**意味理解が他モデルより圧倒的に強い**

---

### 3.5 Phase 1 サマリ

| モデル | Simple | Complex | 評価 |
|--------|--------|---------|------|
| YOLO26 | ❌ 0個検出 | ❌ ガラス板検出せず（spray のみ） | **完敗** |
| DINOv3 | △ 反射部分のみ高類似 | ◯ 板領域がそれっぽく | 補助的 |
| SAM3 | ✅ **板全体マスク 0.84** | ◯ 板含む 3 個（容器も拾う） | **形状◎ / 識別×** |
| Qwen3-VL | ✅ glass plate で完璧 | ✅ **概念で板/容器を区別** | **本実験 MVP** |

**特筆事項**:

1. **YOLO26 は完敗** ─ COCO クラスにガラス板なし、特徴も希薄
2. **Qwen3-VL がディストラクタ識別で唯一勝利** ─ "glass plate" vs "glass" の言語的区別が機能
3. **SAM3 は形状認識最強だが材質識別なし** ─ プロンプト次第で透明容器も拾う

---

## 4. Phase 2 ─ 深度依存モデル × RealSense depth

### 4.1 RealSense IR depth の致命的特性

ガラス板は IR を**透過**するため、RealSense IR depth は **板そのものではなく、板を透過した先の壁面の depth** を返す。

→ depth センサー視点では「板は存在しない」のと等価。これは Phase 2 の根本的限界。

### 4.2 Boxer × RS depth

```bash
python run_boxer.py --input glassboard_simple_rs/ \
  --labels "glass,window,mirror,picture frame,plate,board,sheet" \
  --thresh2d 0.15 --thresh3d 0.15
```

**Simple 結果**:

| クラス | スコア | 位置 (m) | サイズ (cm) |
|--------|-------|---------|------------|
| **mirror** | 0.86 (3D) / 0.21 (2D) | (0.05, 0.34, -0.01) | **18.5 × 8.7 × 16.8** |
| glass (スタンド) | 0.51 | (0.10, 0.35, -0.10) | 5.0 × 6.1 × 5.0 |
| glass (スタンド) | 0.35 | (0.02, 0.46, -0.14) | 5.0 × 20.0 × 5.0 |

実物 14.8×14.8×0.25cm に対し、**横サイズ 18.5cm はおおむね妥当 / 縦サイズ 8.7cm はやや小さい / 厚み 16.8cm は極端に過大**。

![Boxer Simple × RS depth](glassboard/boxer_simple_rs.jpg)
*キャプション: Boxer (Simple, RS depth)。2D OWLv2 でガラス板を **`mirror 0.21`** として bbox 化（COCOにガラス板クラスがないので mirror に近い）。3D OBB は赤い大きな直方体。位置は妥当だが**厚みが17cmと極端に過大**（実物2.5mmなので約70倍）。これは depth が「板を透過して壁面」を見ているため、Boxerが「板の正面 + 壁の奥」を一つのオブジェクト範囲と推定したと考えられる。*

**Complex 結果**:

| クラス | スコア | 位置 (m) | サイズ (cm) | 解説 |
|--------|-------|---------|------------|------|
| bottle | 0.64 | (0.13, 0.36, 0.00) | 7.3 × 7.6 × 17.3 | スプレーボトル |
| glass | 0.71 | (-0.04, 0.37, -0.03) | 6.0 × 6.4 × 11.3 | ペン瓶 |
| glass | 0.63 | (0.04, 0.32, -0.04) | 14.1 × 10.4 × 11.6 | **ガラス板？** |
| glass | 0.58 | (0.10, 0.34, -0.10) | 5.0 × 5.0 × 5.0 | スタンド |
| glass | 0.56 | (0.05, 0.32, -0.09) | 15.5 × 5.0 × 5.0 | スタンド底辺 |

![Boxer Complex × RS depth](glassboard/boxer_complex_rs.jpg)
*キャプション: Boxer (Complex, RS depth)。容器2つ + ガラス板を含む 5 個の OBB を生成。位置精度は妥当だが、ガラス板の OBB 厚みは依然 11.6cm と過大。*

**所見**:
- Boxer の 2D OWLv2 は ガラス板を `mirror` クラスで検出可能
- 3D OBB は **横サイズはおおむね妥当**だが**厚みが極端に過大**（10〜17cm = 実物の40〜70倍）
- これは Boxer のプライアと depth 欠落（板自体に depth なし）の合成結果
- **「板の正面位置」自体は意外と正確**だが、**「板の厚み方向の境界」は推定不能**

### 4.3 Any6D × RS depth

#### マスク・メッシュ生成

| シーン | SAM3 マスク | SF3D メッシュ extents (cm) |
|-------|-----------|------------------------|
| Simple | 0.836 (板全体) | **10.5 × 10.3 × 5.4** |
| Complex | 0.712 (mask=ペン瓶領域) | 9.5 × 5.1 × 1.5 |

実物 14.8×14.8×0.25cm に対し、SF3D は **横サイズ小さめ / 厚み 5.4cm 過剰**。SF3D の透明物体に対する shell 形成癖が出ている。

#### Any6D ポーズ推定

![Any6D Simple × RS depth](glassboard/any6d_simple_rs_pose.jpg)
*キャプション: Any6D (Simple, RS depth)。赤い点群（メッシュ頂点）が**ガラス板の輪郭にぴったり重なる**。Z = 45.0cm、Any6D scale 補正 [1.20, 1.25, 0.83] で SF3D メッシュを伸縮しつつ整合させた結果。*

![Any6D Complex × RS depth](glassboard/any6d_complex_rs_pose.jpg)
*キャプション: Any6D (Complex, RS depth)。**ペン瓶を捕捉してしまっている** (Z=37.0cm)。原因は SAM3 マスク生成時に `glass` プロンプトでペン瓶をマスクしたため。Any6D は与えられたマスクに忠実なので、**入力マスクの誤りを引きずる**。*

**所見**:
- Simple では Any6D が**意外と良好**な結果（メッシュ投影が板にフィット）
- ただし**位置 Z=45cm は実距離より遠い**。RS depth が壁面（板の奥）を測距しているため引きずられた可能性
- Complex では **マスク誤導でペン瓶を捕捉**。これは Phase 1 SAM3 の材質識別不可問題と同根

---

## 5. Phase 1 + 2 中間サマリ

### 5.1 各モデルの能力マトリクス

| モデル | RGB上の検出 | 3D 位置精度 | 厚み推定 | 容器との区別 |
|--------|-----------|----------|--------|----------|
| YOLO26 | ❌ | — | — | — |
| DINOv3 | △ | — | — | — |
| SAM3 | ✅ Simple完璧 | — | — | ❌ |
| Qwen3-VL | ✅ | — | — | ✅ **唯一** |
| Boxer × RS | ◯ (mirror) | △ | ❌ 17cm過大 | △ |
| Any6D × RS | △ | △ | ❌ 5cm過大 | ❌ マスク依存 |

### 5.2 ガラス板特有の難しさ

3glass (容器) との比較：

| 観点 | 透明グラス（容器） | ガラス板（薄板） |
|------|-----------------|------------------|
| YOLO26 検出率 | ✅ 3個 cup | ❌ 0 |
| SAM3 マスク | ◯ プロンプトに敏感 | ✅ 全体マスク |
| Qwen3-VL | グラス3個正確 | **板/容器の区別**が新たな勝利点 |
| RS depth 欠損 | ガラス領域で欠損 | **板を透過、壁面が取れる**（特殊） |
| Boxer 3D OBB | ほぼ妥当 | **厚みが過大** |
| Any6D 失敗モード | スコア低下 | **マスク誤指定でディストラクタ捕捉** |

→ ガラス板は **「透過センサー特性」と「材質識別」の両方が問われる**最難関ケース。

---

## 6. Phase 3 ─ Foundation-Stereo

### 6.1 FS 出力

![FS vis Simple](glassboard/fs_vis_simple.png)
*キャプション: Foundation-Stereo (Simple)。**ガラス板の左右の縁だけ**が薄い縦線として復元され、**面の大半は壁面と同じ深度**として処理。FS でも "板の面" は視差で捕捉できない（透過率が高すぎて 2nd surface が見えない）。*

![FS vis Complex](glassboard/fs_vis_complex.png)
*キャプション: Foundation-Stereo (Complex)。ペン瓶・スプレーボトルは滑らかに復元。**ガラス板自体はほぼ消失**、底のスタンドだけ薄く浮かぶ。*

### 6.2 RGB視点ワープ後の比較

| シーン | RS depth 欠損率 | FS warped 欠損率 | 改善 |
|-------|-------------|-----------------|-----|
| simple | 5.7% | **0.6%** | -5.1pt |
| complex | 16.1% | **0.8%** | -15.3pt |

![Depth比較 Simple](glassboard/depth_comparison_simple.jpg)
*キャプション: Simple 深度比較。RS は壁面ほぼ全面測定（板を透過）、FS warped は **板の左右の縁** だけ薄い線として浮かぶ。**面は依然壁と同じ深度**。*

![Depth比較 Complex](glassboard/depth_comparison_complex.jpg)
*キャプション: Complex 深度比較。RS は容器・板で大量欠損、FS warped はペン瓶・スプレーが滑らかに復元。**ガラス板は底面スタンドのみ可視化**、板の面は引き続き不可視。*

### 6.3 重要な発見

**ガラス容器との大きな違い**:
- グラス（容器）: FS は前面・背面の 2層を分離できた
- **ガラス板（薄板）**: FS でも面は不可視、**エッジしか拾えない**

→ 透過率が高すぎる平板は、ステレオ視差ベースのアルゴリズムでは原理的に困難。前後で同じ画像が見えてしまうため。

---

## 7. Phase 4 ─ FS depth で Boxer / Any6D 再実行

### 7.1 Boxer × FS depth

| シーン | クラス | スコア | 位置 (m) | サイズ (cm) |
|-------|-------|------|---------|------------|
| simple | mirror | 0.87 | (0.05, 0.34, -0.02) | 18.2 × 8.7 × 16.8 |
| simple | glass (スタンド) | 0.54 | (0.10, 0.35, -0.10) | 5.0 × 5.5 × 5.0 |
| complex | bottle (スプレー) | 0.64 | (0.13, 0.36, 0.00) | 7.8 × 7.8 × 17.5 |
| complex | glass (ペン瓶) | 0.71 | (-0.04, 0.37, -0.04) | 6.0 × 6.3 × 11.1 |
| complex | glass (板？) | 0.63 | (0.04, 0.31, -0.04) | 13.3 × 10.4 × 11.1 |

→ RS版とほぼ同じ。**Boxer の厚み推定問題は FS でも解消されない**（FS でも板の面が depth に出ないため）。

![Boxer Simple FS](glassboard/boxer_simple_fs.jpg)
*キャプション: Boxer (Simple, FS depth)。RS版と視覚的にほぼ同等の3D OBB。FSの恩恵は限定的。*

![Boxer Complex FS](glassboard/boxer_complex_fs.jpg)
*キャプション: Boxer (Complex, FS depth)。容器2 + 板 + スタンドの計5 OBB。位置精度 ◯、厚み精度 ✗。*

### 7.2 Any6D × FS depth

![Any6D Simple FS](glassboard/any6d_simple_fs_pose.jpg)
*キャプション: Any6D (Simple, FS depth)。**緑の点群がガラス板にぴったりフィット**。Z = 44.4cm、Any6D scale 補正 [1.11, 1.18, **0.66**] → **厚み方向を 66% に圧縮**（FS depthで板を薄く認識した影響）。RS版（45.0cm, 赤）からわずかに改善。*

![Any6D Complex FS](glassboard/any6d_complex_fs_pose.jpg)
*キャプション: Any6D (Complex, FS depth)。依然**ペン瓶を捕捉**（Z = 36.2cm、RS版 37.0cm とほぼ同じ）。**SAM3 マスクの誤誘導問題は FS化では解消されない**（マスク生成段階で容器に当たっているため）。*

**所見**:
- **Simple Any6D は RS でも FS でも板にフィット**（depth が "壁面" にせよ "板" にせよ大差ないため）
- **Complex Any6D は容器を捕捉し続ける** ─ SAM3 マスクの問題で深度ソースは無関係
- **Boxer の厚み推定は FS でも改善しない** ─ 板の面が depth に出ない以上、根本的な限界

---

## 8. Phase 5 ─ ASGrasp

### 8.1 grasp 検出結果

![ASGrasp Simple](glassboard/asgrasp_simple/grasps_overlay.jpg)
*キャプション: ASGrasp (Simple)。Top 20 grasps。**最高スコア 0.47** でガラス板の底辺スタンド付近、その他は画像の縁・隅にばらつき。**ガラス板自体には把持候補がほぼ集中しない**。*

![ASGrasp Complex](glassboard/asgrasp_complex/grasps_overlay.jpg)
*キャプション: ASGrasp (Complex)。Top 20 grasps が**ペン瓶の上部**に集中（最高 0.44）。スプレーボトルにもいくつか。**ガラス板はスキップ**。*

### 8.2 ASGrasp の限界

- ASGrasp の学習データ (**DREDS**) は主にグラス・ボトル等の**容器系透明物体**で、薄板系は少ない
- 板状物体は**「掴める形状」と判定されない**（grippers が両側から挟む対象として板が薄すぎる、グリッパー幅 8cm に対し 2.5mm なので物理的に "掴めない"）
- → 板状透明物体には ASGrasp は不向き

---

## 9. Phase 6 ─ 単眼深度推定 & FoundationPose × CAD

### 9.1 ReFlow6D を見送った理由

ガラス板は **148×148×2.5mm の単純直方体** なので CAD は容易に生成可能：

```python
import trimesh
mesh = trimesh.creation.box(extents=[0.148, 0.148, 0.0025])  # 148×148×2.5mm
mesh.export("data/glassboard_cad/glass_plate.obj")
```

しかし ReFlow6D は **公式 pretrained 重みが未公開**、Issues での外部リクエストも未回答。
さらに **instance-level の per-object 学習（2201 epoch）** を要し、クラウド GPU でも 2〜3 日コース。
→ 本実験では **ReFlow6D を断念**し、代替として以下 3 系統を評価する：

1. **Depth Anything V2 (Indoor Metric Large)** ─ 単眼 RGB から metric depth を直接推定
2. **Marigold (depth-v1-1)** ─ 拡散モデルベースの単眼相対 depth
3. **FoundationPose × parametric CAD** ─ 直方体 CAD + RGB + depth + マスクで 6DoF ポーズ

---

### 9.2 Depth Anything V2 ─ Indoor Metric Large

**実行**:
```python
from transformers import pipeline
pipe = pipeline("depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", device=0)
out = pipe(rgb_pil)
depth_meters = out["predicted_depth"].squeeze().cpu().numpy()
```

![DA2 Simple](glassboard/depth_anything_v2_simple.jpg)
*キャプション: Depth Anything V2 (Simple)。**ガラス板の "面" が初めて密に depth として復元**された（FS/RS では縁しか取れなかった）。形状は妥当。ただし **metric 値は実距離 ~30cm に対し ~1.5m を出力**（5倍程度の過大推定）。室内学習プライアと、ガラス越しに背景特徴を見ているための混乱が原因と推察。*

![DA2 Complex](glassboard/depth_anything_v2_complex.jpg)
*キャプション: Depth Anything V2 (Complex)。ペン瓶・スプレー・ガラス板を **すべて連続的な depth として復元**。前後関係も妥当。ただし metric 値は同様に実距離より遠く出る。*

**所見**:
- **形状面では本実験初の成果**：板の面そのものを密に復元できた唯一の手法
- **Metric scale はガラス透過の影響で大幅にずれる**（DA2 ~5x 過大）
- → RS の壁面 depth で scale calibration すれば実用可能性あり

---

### 9.3 Marigold (depth-v1-1)

**実行**:
```python
from diffusers import MarigoldDepthPipeline
pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1",
    variant="fp16", torch_dtype=torch.float16).to("cuda")
out = pipe(rgb_pil, num_inference_steps=10, ensemble_size=5)
depth_rel = out.prediction.squeeze().cpu().numpy()  # 相対値 [0,1]
```

![Marigold Simple](glassboard/marigold_simple.jpg)
*キャプション: Marigold (Simple)。**DA2 と同様にガラス板の面を完全に depth として復元**。境界も鋭く、形状品質は本実験最高クラス。ただし **出力は相対 depth のみで metric 値はない**ため、絶対距離・絶対サイズの取得には別途 scale 推定が必要。*

![Marigold Complex](glassboard/marigold_complex.jpg)
*キャプション: Marigold (Complex)。3 物体すべてに加えてテーブル・壁の前後関係まで滑らかに表現。**形状品質は DA2 を上回る**印象だが metric ではない点は同じ。*

**所見**:
- 形状品質は DA2 と同等以上、境界がより鋭い
- Metric scale は **そもそも出力されない**（相対 depth のみ）
- 単独では 6DoF パイプラインに直結できない → 後段で必ず scale 推定が必要

---

### 9.4 FoundationPose × parametric CAD

**入力**:
- CAD: `data/glassboard_cad/glass_plate.obj` (148×148×2.5mm box)
- RGB / depth / mask: Phase 2 で使用したものと同一
- Depth ソース 2 種 × シーン 2 種 = 4 ケース

**実行コマンド**:
```bash
docker exec foundationpose_build bash -c "
  cd /home/vr01/AIVisionPJ/models/foundationpose/FoundationPose &&
  source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
  export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:\$LD_LIBRARY_PATH &&
  QT_QPA_PLATFORM=offscreen python run_demo.py \
    --mesh_file /home/vr01/AIVisionPJ/data/glassboard_cad/glass_plate.obj \
    --test_scene_dir /home/vr01/AIVisionPJ/data/glassboard_${dir} \
    --debug 1 --debug_dir /home/vr01/AIVisionPJ/results/glassboard/fp_${tag}"
```

**結果（4ケースすべて）**:

| ケース | depth | 推定 Z [m] | 観察 |
|-------|------|----------|------|
| simple_rs | RS | 0.423 | CAD が **床に寝かされ 90° 回転誤差**（板は立てているのに） |
| complex_rs | RS | 0.421 | **ガラス板でなく手前のペン瓶位置** に CAD が貼り付く |
| simple_fs | FS | 0.484 | CAD が左下に台形変形した姿で配置、板と非整合 |
| complex_fs | FS | 0.423 | complex_rs と同様、ペン瓶上に CAD |

![FoundationPose Simple × RS](glassboard/foundationpose_simple_rs.jpg)
*キャプション: FoundationPose × CAD (Simple, RS depth)。CAD は板でなく **床に水平に寝た姿勢**で配置される。原因: depth ICP が「板を透過した壁面」に引っ張られ、面を持つ平面探索が床面と整合してしまうため。*

![FoundationPose Complex × RS](glassboard/foundationpose_complex_rs.jpg)
*キャプション: FoundationPose × CAD (Complex, RS depth)。CAD はガラス板でなく **ペン瓶の位置** に配置（マスクは板を指していても、depth 上の最尤平面がペン瓶側にあるため）。*

![FoundationPose Simple × FS](glassboard/foundationpose_simple_fs.jpg)
*キャプション: FoundationPose × CAD (Simple, FS depth)。CAD が左下に台形変形した姿で配置、板と非整合。FS depth でも板の面が取れない以上、ICP の収束先は安定しない。*

![FoundationPose Complex × FS](glassboard/foundationpose_complex_fs.jpg)
*キャプション: FoundationPose × CAD (Complex, FS depth)。complex_rs と同様、CAD はペン瓶上に配置される。*

**所見**:
- **CAD があっても 4/4 で失敗**：板にフィットせず、床面 / 容器 / 不定姿勢へ収束
- 根本原因: FoundationPose の depth refinement (ICP相当) は **観測 depth と CAD 表面の整合**を最大化するが、**観測 depth に「板の面」が存在しない**ため、ICP は次に整合する平面（壁・床・容器）に引っ張られる
- → **CAD を与えても、観測 depth に物体面が無ければ FoundationPose は救えない**
- これは **Phase 2/4 の Boxer / Any6D と同じ根本限界**を改めて確認した結果

---

### 9.5 Phase 6 まとめ

| モデル | 形状認識 | metric 寸法 | 6DoF ポーズ | 位置 |
|--------|---------|-----------|-----------|------|
| **Depth Anything V2** | ✅ **本実験初、板の面を密に復元** | ✗ ~5x 過大 | — (depth 推定器のため出力しない) | — |
| **Marigold** | ✅ DA2 同等以上、境界鋭い | ✗ 相対値のみ | — (同上) | — |
| **FoundationPose × CAD** | — (CAD 既知) | — (CAD 既知) | ✗ 4/4 失敗 | ✗ 床/容器/不定姿勢 |

**重要な理解**:
- DA2 / Marigold は **形状（depth map）の "ピース"** を埋めるのに有効。**6DoF / 3D OBB は出さない**
- 単独でロボット制御に使えるわけではなく、**後段の Boxer や FoundationPose と組み合わせる**ためのコンポーネント
- DA2 metric scale ずれは RS 壁面 depth で校正すれば緩和可能 → **DA2 + RS calibration → FoundationPose** のハイブリッド構成が次の試行候補
- FoundationPose × CAD の失敗は「**観測 depth に物体面が無いと CAD があっても解けない**」という Phase 2/4 と同じ結論を再確認

---

## 10. Phase 7 ─ 透明物体に効く最新単眼 depth × FoundationPose ハイブリッド

### 10.1 動機と評価対象

Phase 6 で得られた知見:
- **DA2 / Marigold は板の "面" を捕捉できる** が metric scale が大きく外れる (DA2 ~5x 過大、Marigold は相対のみ)
- **FoundationPose × CAD は観測 depth に板の面が無いと CAD があっても解けない** (Phase 6 は RS / FS depth で 4/4 失敗)

→ 板面を **metric で** 復元できる単眼 depth モデルが揃えば、FoundationPose × CAD 失敗の根本原因 (depth ICP が壁/床/容器に引っ張られる) を解消できる可能性がある。Phase 7 ではこれを検証する:

| モデル | 種別 | 期待される効果 |
|--------|------|---------------|
| **Metric3D v2 (ViT-Large)** | metric | 法線同時推定、SLAM 統合実績 |
| **MoGe-2 (ViT-L + normal)** | affine-invariant + FoV | 点群・法線・depth 一括取得 |
| **Depth Pro (Apple)** | metric | 境界精度最高クラス |
| **UniDepth V2 (ViT-L)** | metric | 不確実性出力あり |

### 10.2 Complex シーン用「板マスク」の更新

Phase 1〜6 で使用した `mask.png` は Complex シーンでは SAM3 `glass` プロンプトの最高スコア = ペン瓶領域だったため、板評価には不適切だった (Phase 2 / 4 / 6 の Complex 失敗の主因の一つ)。Phase 7 では:

- SAM3 を `glass` プロンプトで実行 → 3 マスク取得 (板・ペン瓶・スプレーボトル)
- **マスク面積最大** (= 板) を採用
- 結果を `mask_plate.png` として保存し、Phase 7 の全モデルでこちらを使用

![Phase 7 Complex 板マスク](glassboard/plate_mask_complex.jpg)
*キャプション: Phase 7 用に再生成した Complex シーンの板マスク。緑領域がガラス板の前面 (ペン瓶・スプレーボトルは除外)。Phase 6 までは `glass` プロンプト最高スコア (= ペン瓶 0.71) を採用していたため、Complex の評価がディストラクタを掴んでいた。*

### 10.3 共通パイプライン

各モデル共通の処理 (`scripts/phase7_<model>.py`):

1. RGB (640×480) → モデル → raw depth (m or affine-invariant)
2. **RS depth の壁面 (板マスク以外、有効値のみ)** で `s = median(rs) / median(model)` の scale 校正
3. 校正後 depth を `data/glassboard_{simple,complex}_fp_<tag>/depth/000000.png` (uint16 mm) に保存
4. FoundationPose × `glass_plate.obj` (148×148×2.5mm box CAD) で 6DoF 推定
5. `results/glassboard/foundationpose_<tag>_{simple,complex}.jpg` に可視化

### 10.4 Metric3D v2 (ViT-Large)

**コマンド**:
```bash
python scripts/phase7_metric3d.py
```
重み: `JUGGHM/Metric3D` (HF), 1.5GB

![Metric3D Simple](glassboard/metric3d_simple.jpg)
*キャプション: Metric3D v2 (Simple)。Raw depth (左下) で**板の面が背景の壁と分離されない**。壁は ~1.10〜1.25m で連続的、板領域は壁とほぼ同値で独立surface として捕捉できていない。校正後 depth (中下) も板領域の median 0.43m = 壁のRS値とほぼ同じ。**Metric3D は板を透過して見ている**。*

![Metric3D Complex](glassboard/metric3d_complex.jpg)
*キャプション: Metric3D v2 (Complex)。ペン瓶・スプレーボトルは raw depth で明確に分離されているが、**板そのものは壁に溶け込んでいる**。校正後 plate-region median 0.42m ≒ RS plate-region 0.42m。*

| シーン | scale | plate median (m) | RS plate median (m) | 所見 |
|-------|-------|-----------------|--------------------|------|
| Simple | 0.353 | 0.433 | 0.419 | **板を透過、壁面 depth と同値** |
| Complex | 0.650 | 0.424 | 0.418 | 同上、容器のみ分離 |

### 10.5 MoGe-2 (ViT-L + normal)

**コマンド**:
```bash
python scripts/phase7_moge2.py
```
重み: `Ruicheng/moge-2-vitl-normal` (HF, 2025-09 公開), ~1.4GB
注意: `fov_x` の単位は **degrees** (誤って radians を渡すと焦点距離が誤推定され depth が ~63m offset で出る → degrees に修正後正常動作)

![MoGe-2 Simple](glassboard/moge2_simple.jpg)
*キャプション: MoGe-2 (Simple)。Raw depth で**板の面が壁よりわずかに前に独立した surface として現れる** (板 ~0.88〜0.96 vs 壁 ~1.05〜1.13、相対的に板が手前)。校正後 plate median 0.36m vs RS plate median (= 壁) 0.42m → **物理的に正しく板を壁の前に配置**。*

![MoGe-2 Complex](glassboard/moge2_complex.jpg)
*キャプション: MoGe-2 (Complex)。ペン瓶・スプレーボトル・板すべてが連続的に分離。校正後 plate region のヒストグラムは 0.34〜0.40m に分布 (RS は 0.42m に集中) → 板を約 2〜6cm 壁より前に配置。*

| シーン | scale | plate median (m) | RS plate median (m) | 所見 |
|-------|-------|-----------------|--------------------|------|
| Simple | 0.394 | 0.361 | 0.419 | **板を壁より約 6cm 前に配置 (物理的に正しい)** |
| Complex | 0.730 | 0.395 | 0.418 | 板が壁より約 2cm 前 |

### 10.6 Depth Pro (Apple)

**コマンド**:
```bash
python scripts/phase7_depthpro.py
```
重み: `apple/DepthPro` (HF), 1.9GB

![Depth Pro Simple](glassboard/depthpro_simple.jpg)
*キャプション: Depth Pro (Simple)。Raw depth で**板が極めてシャープな矩形** (~1.13m) として浮かび上がる。境界がピクセルレベルで鋭く、本実験で最も明瞭な板の depth 表現。校正後 plate median 0.37m vs RS plate (壁) 0.42m。*

![Depth Pro Complex](glassboard/depthpro_complex.jpg)
*キャプション: Depth Pro (Complex)。容器の凹凸も板の輪郭も極めて鮮明。校正後 plate region 0.34m。**境界のシャープネスが抜きん出ている**。*

| シーン | scale | plate median (m) | RS plate median (m) | 所見 |
|-------|-------|-----------------|--------------------|------|
| Simple | 0.326 | 0.374 | 0.419 | **境界最高クラス、板を約 5cm 前に配置** |
| Complex | 0.785 | 0.344 | 0.418 | 同様に鋭い |

### 10.7 UniDepth V2 (ViT-L)

**コマンド**:
```bash
python scripts/phase7_unidepth.py
```
重み: `lpiccinelli/unidepth-v2-vitl14` (HF)
特徴: depth と同時に **confidence map** を出力

![UniDepth V2 Simple](glassboard/unidepth_simple.jpg)
*キャプション: UniDepth V2 (Simple)。Raw depth で板が壁よりわずかに前に分離 (~0.95〜1.00 vs 壁 ~1.05〜1.08)、境界は Depth Pro より softer。**右上の confidence map** で板の輪郭領域が低信頼度 (青〜紫) として表現されており、後段で重み付けに使える。*

![UniDepth V2 Complex](glassboard/unidepth_complex.jpg)
*キャプション: UniDepth V2 (Complex)。容器のシルエットは Depth Pro 同等にシャープだが、板そのものは壁との分離が弱い。confidence は容器側で高、板領域で低。*

| シーン | scale | plate median (m) | RS plate median (m) | plate conf (median) | 所見 |
|-------|-------|-----------------|--------------------|---------------------|------|
| Simple | 0.397 | 0.397 | 0.419 | 0.64 | 板を約 2cm 前、信頼度中 |
| Complex | 0.470 | 0.422 | 0.418 | 0.61 | 板の分離弱、信頼度中 |

### 10.8 FoundationPose × parametric CAD (depth 置換)

校正後 depth を Phase 6 と同様 `data/glassboard_{simple,complex}_fp_<tag>/depth/000000.png` (uint16 mm) として保存し、`docker exec foundationpose_build python run_demo.py --mesh_file glass_plate.obj ...` で実行。

**実行結果 (4 モデル × 2 シーン = 8 ケース)**:

![FP × Metric3D Simple](glassboard/foundationpose_metric3d_simple.jpg)
*キャプション: FP × Metric3D (Simple)。**CAD が板の底辺で水平に寝た姿勢**。Metric3D が板を独立 surface として捉えていないため Phase 6 RS と同じ失敗モード。Z=0.401m。*

![FP × Metric3D Complex](glassboard/foundationpose_metric3d_complex.jpg)
*キャプション: FP × Metric3D (Complex)。**CAD が板の中央位置で正立**。容器を回避してピンポイントで板を捕捉 (Phase 6 ではペン瓶上だった)。マスク改善 + 容器 depth が効いている。Z=0.432m。*

![FP × MoGe-2 Simple](glassboard/foundationpose_moge2_simple.jpg)
*キャプション: FP × MoGe-2 (Simple)。**CAD がガラス板に正しい位置・サイズ・正立姿勢で配置**。Phase 6 で 0/4 だった成功ケース第一号。Z=0.365m。*

![FP × MoGe-2 Complex](glassboard/foundationpose_moge2_complex.jpg)
*キャプション: FP × MoGe-2 (Complex)。CAD は容器の間で板を捕捉、姿勢は正立。サイズが板実物より小さめだが位置は妥当。Z=0.453m。*

![FP × Depth Pro Simple](glassboard/foundationpose_depthpro_simple.jpg)
*キャプション: FP × Depth Pro (Simple)。**CAD が板の輪郭にほぼ完全フィット**。本実験の到達点。Z=0.372m。*

![FP × Depth Pro Complex](glassboard/foundationpose_depthpro_complex.jpg)
*キャプション: FP × Depth Pro (Complex)。**CAD が板の正しい位置・姿勢で配置**、容器の手前にも回り込まない。サイズも実物大。Z=0.352m。*

![FP × UniDepth V2 Simple](glassboard/foundationpose_unidepth_simple.jpg)
*キャプション: FP × UniDepth V2 (Simple)。CAD が板に正立姿勢で重なるが、右側に少しオフセット。姿勢・サイズは妥当。Z=0.399m。*

![FP × UniDepth V2 Complex](glassboard/foundationpose_unidepth_complex.jpg)
*キャプション: FP × UniDepth V2 (Complex)。CAD は板領域に正立、容器を回避。やや左寄り。Z=0.426m。*

**FP 結果サマリ**:

| ケース | Z [m] | CAD 整合度 | 観察 |
|-------|------|------|------|
| metric3d × simple | 0.401 | ✗ | CAD が水平に寝た姿勢、Phase 6 RS と同じ失敗 |
| metric3d × complex | 0.432 | ✅ | 板位置に正立、Phase 6 の容器捕捉から改善 |
| moge2 × simple | 0.365 | ✅✅ | 板に正しい位置・姿勢、サイズも妥当 |
| moge2 × complex | 0.453 | ✅ | 板位置に正立、サイズ小さめ |
| **depthpro × simple** | 0.372 | ✅✅✅ | **本実験の到達点、板にほぼ完全フィット** |
| **depthpro × complex** | 0.352 | ✅✅✅ | **同様に Complex でも完全フィット** |
| unidepth × simple | 0.399 | ✅ | 正立姿勢、右にオフセット |
| unidepth × complex | 0.426 | ✅ | 正立姿勢、左にオフセット |

**成功率 7/8** (Phase 6 は 0/4)。

### 10.9 Phase 7 まとめ

| モデル | 板の面を depth 化 | metric scale | 境界鋭さ | FP × CAD 成功 | 特記 |
|--------|---------------|------------|--------|-------------|------|
| Metric3D v2 (ViT-L) | ✗ 壁と一体化 | ◯ (校正前提) | △ | 1/2 (Complex のみ) | 板透過、容器のみ分離 |
| MoGe-2 (ViT-L + normal) | ✅ 壁前に配置 | ◯ (校正前提) | ◯ | 2/2 | 法線・点群も同時取得 |
| **Depth Pro** | ✅ 鮮明な矩形 | ◯ (校正前提) | ✅ **本実験最高** | **2/2** | **総合トップ、Simple/Complex とも完全フィット** |
| UniDepth V2 (ViT-L) | ◯ 板やや弱い | ◯ (校正前提) | △ | 2/2 | confidence map が独自の付加価値 |

**重要な発見**:

1. **板面を独立 surface として depth 化できるか否かが分水嶺**
   - Metric3D v2 は metric SOTA だが「ガラスを透過して背景を見る」プライアが強く、**板面を捕捉できない**
   - MoGe-2 / Depth Pro / UniDepth V2 は **板面を壁より前に独立 surface として復元** → FP の ICP がそこに収束し pose が解ける
   - DA2 (Phase 6) も板面は捕捉できていたが metric scale が ~5x 過大で校正が困難

2. **scale 校正は RS depth の壁面 1点で十分**
   - 全モデルで「板マスクを除外した RS 有効領域の median ratio」というシンプルな校正で metric 整合
   - 板を透過した depth (= 壁面 RS depth) を基準に使うのは「板そのものの基準距離が無い」状況下で唯一実用的

3. **Complex シーンの "板マスク" 修正が重要**
   - Phase 1 SAM3 `glass` プロンプトで最高スコアを取ると Complex ではペン瓶 (0.71) が選ばれていた
   - Phase 7 では「**最大面積マスク = 板**」基準に切替 (面積 62558 px > ペン瓶 19744 px > スプレー 25355 px)
   - これだけで Phase 6 の Metric3D Complex 失敗 (CAD がペン瓶上に貼り付く) も改善した

4. **Depth Pro は境界鋭さで頭一つ抜けている**
   - 1536² の高解像度処理 + Apple 独自の boundary refinement
   - VRAM ~6GB で本 PC でも余裕で動作
   - **現時点でガラス板用の最有力単眼 depth 候補**

5. **UniDepth V2 の confidence は今後の道**
   - 板領域は中信頼度 (0.61〜0.64)、容器・壁は高信頼度
   - 後段で「信頼度に応じた depth 加重」を行えばさらにロバスト化可能 (Phase 7 では未活用)

### 10.10 結論

- **ガラス板の 6DoF ポーズが初めて FoundationPose × CAD で解けた** (Phase 6 の 0/4 から 7/8)
- **Depth Pro が現時点のベスト**: Simple / Complex とも完全フィット、境界精度最高
- MoGe-2 は法線・点群も同時に取れる利点があり、後段で OBB 構築・把持計画にも展開しやすい
- Metric3D v2 は SOTA metric depth だが「透明物体は背景として扱う」学習バイアスが強く、ガラス板用途には不向き
- ハイブリッド構成 **「Qwen3-VL/SAM3 でマスク → Depth Pro / MoGe-2 で板面 depth → RS壁面で校正 → FoundationPose × parametric CAD」** がガラス板向け 6DoF パイプラインの実用解として確立

---

## 11. 全体サマリ（Phase 1〜7）

### 11.1 各モデルの能力マトリクス（最終版）

| モデル | RGB上の検出 | 形状(depth) | 3D 位置精度 | 厚み推定 | 6DoFポーズ | 容器との区別 | 把持可能性 |
|--------|-----------|-----------|----------|--------|----------|----------|-----------|
| YOLO26 | ❌ | — | — | — | — | — | — |
| DINOv3 | △ | — | — | — | — | — | — |
| SAM3 | ✅ Simple完璧 | — | — | — | — | ❌ | — |
| Qwen3-VL | ✅ | — | — | — | — | ✅ **唯一** | — |
| Boxer × RS | ◯ (mirror) | — | △ | ❌ 17cm過大 | △ | △ | △ |
| Any6D × RS | △ | — | ◯ Simple | ❌ 5cm過大 | ◯ Simple | ❌ マスク依存 | △ |
| Boxer × FS | ◯ (mirror) | — | △ | ❌ 同様 | △ | △ | △ |
| Any6D × FS | ✅ Simple | — | ◯ Simple | ◯ FS で薄め補正 | ◯ Simple | ❌ マスク依存 | △ |
| ASGrasp | — | — | — | — | — | — | **❌ 板を grasp 対象とせず** |
| **DA2 (Indoor Metric)** | — | ✅ **面を復元** | ✗ 5x 過大 | △ | — | — | — |
| **Marigold** | — | ✅ **DA2同等以上** | — (相対) | — | — | — | — |
| **FP × CAD (Phase 6, RS/FS)** | — | — | ✗ | — | **✗ 4/4失敗** | — | — |
| **Metric3D v2 (Phase 7)** | — | ✗ 板透過 | ◯ 校正後 | — | △ 1/2 | — | — |
| **MoGe-2 (Phase 7)** | — | ✅ 板を前に配置 | ◯ 校正後 | — | ✅ 2/2 | — | — |
| **Depth Pro (Phase 7)** | — | ✅ **境界最鮮明** | ◯ 校正後 | — | ✅ **2/2** | — | — |
| **UniDepth V2 (Phase 7)** | — | ◯ 板分離弱 | ◯ 校正後 + conf | — | ✅ 2/2 | — | — |
| **FP × CAD (Phase 7, depth置換)** | — | — | ◯ | — | ✅ **7/8 成功** | — | — |

### 11.2 ガラス板 vs ガラス容器（前回 3glass 実験との対比）

| 観点 | 透明グラス（容器） | ガラス板（薄板） |
|------|-----------------|------------------|
| YOLO26 | ✅ 3個 cup | ❌ 0 |
| SAM3 | ◯ プロンプト依存 | ✅ Simple ほぼ完璧 |
| Qwen3-VL | グラス3個 | **板/容器の言語的区別**が新たな勝利点 |
| RS depth 欠損 | ガラス領域で欠損 | **板を透過、壁面が取れる**（特殊） |
| FS depth | 2層分離成功 | **エッジのみ復元、面は不可視** |
| Boxer 3D OBB | ほぼ妥当 | **厚みが過大** |
| Any6D ポーズ | 改善 (RS→FS) | Simple◯、Complex はマスク依存 |
| ASGrasp grasp | ✅ Top 0.74 で側面把持 | **❌ 把持候補が板に集中しない** |

### 11.3 結論

- **ガラス板は容器より深刻に難しい**（RGB特徴希薄 + depth 透過 + 把持困難）
- **Phase 1 では Qwen3-VL がディストラクタ識別で唯一勝利**（言語的概念で「板」と「容器」を区別）
- **Phase 2〜4 では Boxer/Any6D とも厚み推定の限界**が明確
- **Phase 5 ASGrasp は薄板を把持対象とせず**（DREDSの学習バイアス + 物理的制約）
- **Phase 6 で初めて板の "面" を depth として捕捉**（DA2 / Marigold）したが metric scale が破綻、FP × CAD は 4/4 失敗
- **Phase 7 で 6DoF ポーズが初めて解けた** ─ Depth Pro / MoGe-2 / UniDepth V2 は metric な板面 depth を出せる。RS 壁面で scale 校正 → FoundationPose × parametric CAD で **7/8 ケース成功**
- **Depth Pro が現時点のガラス板向け単眼 depth ベスト**: 境界最鮮明、Simple / Complex とも完全フィット
- ReFlow6D は pretrained 重み未公開のため断念。**Phase 7 のハイブリッド構成が実用解**として確立した
- 今後の方向: ① UniDepth V2 confidence を活用した depth 重み付け ② MoGe-2 法線・点群を併用した OBB 構築 ③ 異視点撮影での pose 精度向上

---

## 12. ファイル構成（現時点）

```
data/glassboard_{simple,complex}/
├── rgb.png                      入力RGB (Phase 1-7 共通)
├── left_ir.png / right_ir.png   IR ステレオ (Phase 3, 5 用)
├── depth.npy / depth.png        RS depth (Phase 2, 6-7 校正用)
├── depth_da2.npy                Depth Anything V2 出力 (Phase 6)
├── depth_marigold.npy           Marigold 出力 (Phase 6)
├── depth_metric3d.npy           Metric3D v2 出力 (Phase 7)
├── depth_moge2.npy              MoGe-2 出力 (Phase 7)
├── depth_depthpro.npy           Depth Pro 出力 (Phase 7)
├── depth_unidepth.npy           UniDepth V2 出力 (Phase 7)
├── intrinsics.json              RGB + IR + extrinsics
├── K_ir.txt                     Foundation-Stereo 互換
├── mask.png                     SAM3 生成マスク (Phase 2, 6 用: Complex はペン瓶)
├── mask_plate.png               Phase 7 用板マスク (Complex: SAM3 最大面積=板)
├── qwen_bbox_plate.json         Qwen3-VL 板 bbox (Phase 7 マスク生成時)
├── mesh.obj                     SF3D 生成メッシュ (Phase 2 用)
└── preview.jpg

data/glassboard_cad/
└── glass_plate.obj              148×148×2.5mm parametric box CAD (Phase 6-7)

data/glassboard_{simple,complex}_rs/              Boxer 用 (RS depth + flat intrinsics)
data/glassboard_{simple,complex}_fp{,_fs}/        FoundationPose 用 (Phase 6: RS/FS depth)
data/glassboard_{simple,complex}_fp_metric3d/     FoundationPose 用 (Phase 7: Metric3D depth)
data/glassboard_{simple,complex}_fp_moge2/        FoundationPose 用 (Phase 7: MoGe-2 depth)
data/glassboard_{simple,complex}_fp_depthpro/     FoundationPose 用 (Phase 7: Depth Pro depth)
data/glassboard_{simple,complex}_fp_unidepth/     FoundationPose 用 (Phase 7: UniDepth V2 depth)

results/glassboard/
├── input_{simple,complex}.png
├── capture_preview_{simple,complex}.jpg
├── yolo26_{simple,complex}.jpg                        Phase 1.1
├── dinov3_{simple,complex}_{pca,sim}.jpg              Phase 1.2
├── sam3_{simple,complex}_*.jpg                        Phase 1.3
├── qwen3vl_{simple,complex}_*.jpg                     Phase 1.4
├── boxer_{simple,complex}_{rs,fs}.jpg                 Phase 2.1, 4.1
├── any6d_{simple,complex}_{rs,fs}_pose.jpg            Phase 2.2, 4.2
├── fs_vis_{simple,complex}.png                        Phase 3
├── depth_comparison_{simple,complex}.jpg              Phase 3
├── asgrasp_{simple,complex}/                          Phase 5
├── depth_anything_v2_{simple,complex}.jpg             Phase 6.2
├── marigold_{simple,complex}.jpg                      Phase 6.3
├── foundationpose_{simple,complex}_{rs,fs}.jpg        Phase 6.4
├── fp_{simple,complex}_{rs,fs}/ob_in_cam/             Phase 6 FP ポーズ出力
├── plate_mask_complex.jpg                             Phase 7 板マスク確認図
├── metric3d_{simple,complex}.jpg                      Phase 7.1 depth 可視化
├── moge2_{simple,complex}.jpg                         Phase 7.2 depth 可視化
├── depthpro_{simple,complex}.jpg                      Phase 7.3 depth 可視化
├── unidepth_{simple,complex}.jpg                      Phase 7.4 depth 可視化
├── foundationpose_metric3d_{simple,complex}.jpg       Phase 7 FP 可視化
├── foundationpose_moge2_{simple,complex}.jpg          Phase 7 FP 可視化
├── foundationpose_depthpro_{simple,complex}.jpg       Phase 7 FP 可視化
├── foundationpose_unidepth_{simple,complex}.jpg       Phase 7 FP 可視化
├── fp_{simple,complex}_{metric3d,moge2,depthpro,unidepth}/ob_in_cam/  Phase 7 FP ポーズ
├── {metric3d,moge2,depthpro,unidepth}_summary.json   Phase 7 数値サマリ
└── fp_{simple,complex}_{metric3d,moge2,depthpro,unidepth}/track_vis/  FP 可視化 raw

scripts/
├── phase7_metric3d.py           Phase 7.1 Metric3D v2 runner
├── phase7_moge2.py              Phase 7.2 MoGe-2 runner
├── phase7_depthpro.py           Phase 7.3 Depth Pro runner
├── phase7_unidepth.py           Phase 7.4 UniDepth V2 runner
├── phase7_make_plate_mask_complex.py  Complex 板マスク生成
├── phase7_run_foundationpose.sh Phase 7.5 FP 8 ケース一括実行
└── _phase7_sam3_largest.py      SAM3 最大面積マスク取得ヘルパー
```
