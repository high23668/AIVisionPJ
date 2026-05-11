# 単眼深度推定 技術調査・実験報告
## DA2 / MoGe-2 / Depth Pro / UniDepth V2 / Metric3D v2 / Marigold

**文書番号**: PDX-RB20260504-005  
**作成日**: 2026-05-04  
**作成者**: 室屋  
**関連文書**: [概観レポート](00_overview.md) / [前レポート: VLM](03_vlm.md)

---

## 1. この技術が解決する課題

ロボットが物体を掴むためには「物体までの距離（深度）」が必要である。
透明物体は赤外線センサ（RealSense等）を素通りするため深度が取れない。
**単眼深度推定**は、RGB画像1枚から距離マップを生成するAI技術であり、
センサの制約を回避する手段として期待される。

```
入力: RGB画像 1枚
出力: 深度マップ（各ピクセルに距離値を付与した画像）
```

ただし単眼深度推定にはセンサ深度と異なる特性があり、
そのままでは実距離（メートル）に対応しない場合がある。
この問題と対処法（校正）を含めて解説する。

---

## 2. 評価した6モデルの概要

| # | モデル | 開発元 | 出力タイプ | 推論時間 |
|---|---|---|---|---|
| 1 | **Depth Anything V2 Indoor（DA2）** | ByteDance | metric | ~0.5秒 |
| 2 | **MoGe-2 ViT-L** | Microsoft | affine-invariant | ~0.3秒 |
| 3 | **Depth Pro** | Apple | metric | ~1.0秒 |
| 4 | **UniDepth V2 ViT-L** | Valeo AI | metric | ~0.1秒 |
| 5 | **Metric3D v2 ViT-Large** | HKUST | metric | ~1.3秒 |
| 6 | **Marigold LCM** | ETH Zürich | affine-invariant | ~5〜10秒 |

---

## 3. 重要な前提知識

### 3.1 Metric depth と Affine-invariant depth の違い

単眼深度モデルには出力の種類が2つある。これを理解することが本章全体の鍵となる。

| 出力タイプ | 特徴 | 代表モデル |
|---|---|---|
| **Metric（絶対距離）** | カメラ校正情報を使いメートル単位で出力 | DA2, Depth Pro, UniDepth V2, Metric3D v2 |
| **Affine-invariant（相対距離）** | 「遠い・近い」という相対的な遠近関係のみを出力。スケール・オフセット未定 | MoGe-2, Marigold |

```
Metric の出力例:
  ピクセル(320,240) = 0.383m（そのまま実距離として使える）

Affine-invariant の出力例:
  ピクセル(320,240) = 0.72（相対値。実距離への変換が必要）
```

Affine-invariant モデルは実距離を直接出力しないが、
**形状の遠近関係を正確に捉える**点で優れており、
後述の校正処理で実距離に変換できる。

### 3.2 深度校正の必要性と方式

単眼モデルの出力は誤差を含むため、Ground Truth（実測値）との照合による校正が必要である。

**校正方式の比較:**

| 方式 | 計算 | 必要なもの | 精度 |
|---|---|---|---|
| **壁校正（1点）** | `depth_cal = s × depth_raw`（スケールのみ） | 壁面の実距離 | 低（オフセット未補正） |
| **Affine校正（2点）** | `depth_cal = a × depth_raw + b` | 壁面 + 板面のGT距離 | 高 |

**Affine校正の仕組み:**

```
壁面の実距離（y1）とモデル出力（x1）
板面のGT距離（y2）とモデル出力（x2）の2点から

a（スケール） = (y1 - y2) / (x1 - x2)
b（オフセット） = y1 - a × x1

→ depth_cal = a × depth_raw + b
```

### 3.3 Affine校正崩壊：樹脂シート正面での特殊問題

樹脂シートを正面から撮影した場合（resin_front）、
カメラからシートまでの距離（163mm）と壁までの距離が
**ほぼ同じになる**ことがある。

```
x1（壁のモデル出力）≈ x2（板のモデル出力）のとき

分母 (x1 - x2) → 0
→ a = (y1 - y2) / 0 → 発散（無限大）
```

これが「Affine校正崩壊」である。
各モデルの Affine 校正係数 a の実測値：

| モデル | resin_front の a 値 | 崩壊の有無 |
|---|---|---|
| DA2 | 0.252 | △（b=-52.9mと極端） |
| **MoGe-2** | **1.487** | **✅ 安定** |
| Depth Pro | **-423.6** | ✗ 完全崩壊 |
| UniDepth V2 | 15.4 | ✗ 発散 |
| Metric3D v2 | 23.2 | ✗ 発散 |
| Marigold | 73.6 | ✗ 発散 |

MoGe-2 のみが安定している理由：affine-invariant 設計により
「板と壁の相対的な深度差」を正確に捉えており、
絶対距離が近くても内部表現で両者を区別できるためと考えられる。

---

## 4. 実験設計

### 4.1 評価シーン

| シーン | ワーク | 角度 | GT Z |
|---|---|---|---|
| glass_front | ガラス板 | 正面 | 383.6mm |
| glass_oblique | ガラス板 | 30°斜め | 377.0mm |
| resin_front | 樹脂シート | 正面 | 163.0mm |
| resin_oblique | 樹脂シート | 30°斜め | 175.0mm |

### 4.2 評価指標

- **RMSE（GT校正後）**: 板面ピクセルの推定深度と GT Z の二乗平均平方根誤差（mm）
- **Inlier@2cm / @5cm**: 誤差が ±20mm / ±50mm 以内のピクセル割合

### 4.3 可視化の見方

各モデルの結果画像は4パネル構成になっている。

```
[左上] RGB入力          [右上] モデル生出力（疑似カラー）
[左下] 壁校正後          [右下] GT Affine校正後 + 誤差マップ
                                （白に近いほど誤差小）
```

---

## 5. 全モデル定量結果

### GT Affine校正後 RMSE（mm）

| モデル | glass_front | glass_oblique | resin_front | resin_oblique |
|---|---|---|---|---|
| **DA2** | **19.3** | 28.2 | 1178（崩壊） | 62.2 |
| **MoGe-2** | 20.4 | 36.5 | **33.8** | **27.0** |
| Depth Pro | 20.3 | 28.1 | 38587（崩壊） | 164.9 |
| UniDepth V2 | 56.0 | 146.0 | 172.1 | 195.9 |
| Metric3D v2 | 179.6 | 53974（崩壊） | 309.9 | 113.1 |
| Marigold | 22.0 | **26.2** | 647（崩壊） | 208.2 |

### Inlier@2cm（GT校正後）

| モデル | glass_front | glass_oblique | resin_front | resin_oblique |
|---|---|---|---|---|
| **DA2** | **98.2%** | 52.6% | 9.4% | 71.9% |
| **MoGe-2** | 98.1% | 33.5% | **45.0%** | **91.9%** |
| Depth Pro | 95.8% | 47.7% | 3.1% | 8.2% |
| UniDepth V2 | 70.7% | 11.8% | 10.7% | 9.2% |
| Metric3D v2 | 5.0% | 0.5% | 7.4% | 15.7% |
| Marigold | 97.9% | 60.9% | 1.5% | 10.4% |

**MoGe-2 が唯一、全4シーンで実用的な精度を維持した。**

---

## 6. 各モデル詳細

### 6.1 Depth Anything V2 Indoor（DA2）

**開発**: ByteDance / 2024年  
**論文**: [arxiv 2406.09414](https://arxiv.org/abs/2406.09414) / [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)  
**特徴**: Indoor 特化版。近距離・室内環境の深度推定に最適化されたメトリックモデル。

**ガラス板での最高精度を達成（RMSE 19.3mm）。**

![DA2 glass_front](../exp2/depth/da2_glass_front_vis.jpg)
*DA2 — glass_front。右下の GT Affine校正後の誤差マップがほぼ白（低誤差）。
壁校正後（左下）では板が真っ暗になるが、GT校正後は正しいグリーン帯に修正される。
RMSE=19.3mm、Inlier@2cm=98.2%（全モデル中最高）。*

![DA2 glass_oblique](../exp2/depth/da2_glass_oblique_vis.jpg)
*DA2 — glass_oblique。RMSE=28.2mm。斜め撮影では精度がやや低下。*

![DA2 resin_front](../exp2/depth/da2_resin_front_vis.jpg)
*DA2 — resin_front。Affine校正崩壊（a=0.252, b=-52.9m）によりRMSE=1178mm。
モデル出力では板と背景の深度差がほぼ検出されていない。*

![DA2 resin_oblique](../exp2/depth/da2_resin_oblique_vis.jpg)
*DA2 — resin_oblique。RMSE=62.2mm。斜め撮影では崩壊が起きず実用的な精度。*

| シーン | RMSE | Inlier@2cm | Inlier@5cm |
|---|---|---|---|
| glass_front | **19.3mm** | **98.2%** | 98.4% |
| glass_oblique | 28.2mm | 52.6% | 97.5% |
| resin_front | 1178mm（崩壊） | 9.4% | 9.4% |
| resin_oblique | 62.2mm | 71.9% | 96.2% |

---

### 6.2 MoGe-2 ViT-L

**開発**: Microsoft / 2024年  
**論文**: [arxiv 2507.02546](https://arxiv.org/abs/2507.02546) / [GitHub](https://github.com/microsoft/MoGe)  
**特徴**: Affine-invariant 設計。スケール・オフセット依存なく相対的な形状を正確に捉える。

**全4シーンで唯一崩壊なし。透明ワーク対応の最重要モデル。**

![MoGe-2 glass_front](../exp2/depth/moge2_glass_front_vis.jpg)
*MoGe-2 — glass_front。RMSE=20.4mm。DA2 に次ぐ精度でガラス板の形状を捉えている。*

![MoGe-2 glass_oblique](../exp2/depth/moge2_glass_oblique_vis.jpg)
*MoGe-2 — glass_oblique。RMSE=36.5mm。斜め撮影でも安定した出力。*

![MoGe-2 resin_front](../exp2/depth/moge2_resin_front_vis.jpg)
*MoGe-2 — resin_front。RMSE=33.8mm。Affine校正が安定（a=1.487）。
他モデルが完全崩壊する中、唯一実用的な精度を維持した。*

![MoGe-2 resin_oblique](../exp2/depth/moge2_resin_oblique_vis.jpg)
*MoGe-2 — resin_oblique。RMSE=27.0mm、Inlier@2cm=91.9%。全シーン中最良。*

| シーン | RMSE | Inlier@2cm | Inlier@5cm |
|---|---|---|---|
| glass_front | 20.4mm | 98.1% | 98.4% |
| glass_oblique | 36.5mm | 33.5% | 84.8% |
| resin_front | **33.8mm** | 45.0% | 90.0% |
| resin_oblique | **27.0mm** | **91.9%** | 98.7% |

---

### 6.3 Depth Pro（Apple）

**開発**: Apple / 2024年  
**論文**: [arxiv 2410.02073](https://arxiv.org/abs/2410.02073) / [GitHub](https://github.com/apple/ml-depth-pro)  
**特徴**: カメラ内部パラメータなしで focal length を自動推定してメトリック出力。

**ガラス板には高精度だが、樹脂シートは完全崩壊。**

![Depth Pro glass_front](../exp2/depth/depthpro_glass_front_vis.jpg)
*Depth Pro — glass_front。RMSE=20.3mm。ガラス板の輪郭を鮮明に捉えており
境界最鮮明なモデル（Phase 1〜7 実験で最良と評価）。*

![Depth Pro glass_oblique](../exp2/depth/depthpro_glass_oblique_vis.jpg)
*Depth Pro — glass_oblique。RMSE=28.1mm。*

![Depth Pro resin_front](../exp2/depth/depthpro_resin_front_vis.jpg)
*Depth Pro — resin_front。Affine校正係数 a=-423.6 と完全崩壊。
RMSE=38,587mm。樹脂シートの深度情報がほぼ取れていない。*

![Depth Pro resin_oblique](../exp2/depth/depthpro_resin_oblique_vis.jpg)
*Depth Pro — resin_oblique。崩壊は免れたがRMSE=164.9mmと精度低。*

| シーン | RMSE | Inlier@2cm | Inlier@5cm |
|---|---|---|---|
| glass_front | 20.3mm | 95.8% | 97.9% |
| glass_oblique | 28.1mm | 47.7% | 96.9% |
| resin_front | 38,587mm（崩壊） | 3.1% | 3.3% |
| resin_oblique | 164.9mm | 8.2% | 26.7% |

---

### 6.4 UniDepth V2 ViT-L

**開発**: Valeo AI / 2024年  
**論文**: [arxiv 2403.18913](https://arxiv.org/abs/2403.18913) / [GitHub](https://github.com/lpiccinelli-eth/UniDepth)  
**特徴**: カメラ幾何を学習した高速メトリックモデル。推論 ~130ms と最速クラス。

**速度は最速だが精度は他モデルより劣る。**

![UniDepth glass_front](../exp2/depth/unidepth_glass_front_vis.jpg)
*UniDepth V2 — glass_front。RMSE=56.0mm。ガラス板領域の深度ムラが大きく
板面を均一に捉えられていない。*

![UniDepth glass_oblique](../exp2/depth/unidepth_glass_oblique_vis.jpg)
*UniDepth V2 — glass_oblique。RMSE=146.0mmと大きく劣化。*

![UniDepth resin_front](../exp2/depth/unidepth_resin_front_vis.jpg)
*UniDepth V2 — resin_front。崩壊（a=15.4）によりRMSE=172.1mm。*

![UniDepth resin_oblique](../exp2/depth/unidepth_resin_oblique_vis.jpg)
*UniDepth V2 — resin_oblique。RMSE=195.9mm。全シーン中最も安定性に欠ける。*

| シーン | RMSE | Inlier@2cm | Inlier@5cm |
|---|---|---|---|
| glass_front | 56.0mm | 70.7% | 79.7% |
| glass_oblique | 146.0mm | 11.8% | 27.1% |
| resin_front | 172.1mm | 10.7% | 26.6% |
| resin_oblique | 195.9mm | 9.2% | 22.6% |

---

### 6.5 Metric3D v2 ViT-Large

**開発**: HKUST / 2024年  
**論文**: [arxiv 2404.15506](https://arxiv.org/abs/2404.15506) / [GitHub](https://github.com/YvanYin/Metric3D)  
**特徴**: 大規模データ学習による汎用メトリックモデル。多様なシーンに対応。

**透明物体を「透過して背景と同一視」する傾向があり本ユースケースに不適。**

![Metric3D glass_front](../exp2/depth/metric3d_glass_front_vis.jpg)
*Metric3D v2 — glass_front。RMSE=179.6mm。ガラス板を透過して背景と同じ深度を
割り当てており、板面の深度情報がほぼ取れていない。*

![Metric3D glass_oblique](../exp2/depth/metric3d_glass_oblique_vis.jpg)
*Metric3D v2 — glass_oblique。Affine崩壊でRMSE=53,974mm。*

![Metric3D resin_front](../exp2/depth/metric3d_resin_front_vis.jpg)
*Metric3D v2 — resin_front。RMSE=309.9mm。*

![Metric3D resin_oblique](../exp2/depth/metric3d_resin_oblique_vis.jpg)
*Metric3D v2 — resin_oblique。RMSE=113.1mm。*

| シーン | RMSE | Inlier@2cm | 備考 |
|---|---|---|---|
| glass_front | 179.6mm | 5.0% | 透明透過 |
| glass_oblique | 53,974mm | 0.5% | 崩壊 |
| resin_front | 309.9mm | 7.4% | |
| resin_oblique | 113.1mm | 15.7% | |

---

### 6.6 Marigold LCM

**開発**: ETH Zürich / 2024年  
**論文**: [arxiv 2312.02145](https://arxiv.org/abs/2312.02145) / [GitHub](https://github.com/prs-eth/Marigold)  
**特徴**: 拡散モデル（Stable Diffusion）ベースの深度推定。affine-invariant 出力。
LCM（Latent Consistency Model）で高速化。

**ガラス板は高精度だが処理時間が長く（〜10秒）、樹脂シートは崩壊。**

![Marigold glass_front](../exp2/depth/marigold_glass_front_vis.jpg)
*Marigold LCM — glass_front。RMSE=22.0mm。拡散モデルの生成品質が活き、
ガラス板の細部まで滑らかな深度マップを生成している。*

![Marigold glass_oblique](../exp2/depth/marigold_glass_oblique_vis.jpg)
*Marigold LCM — glass_oblique。RMSE=26.2mm（全モデル中 glass_oblique 最高）。*

![Marigold resin_front](../exp2/depth/marigold_resin_front_vis.jpg)
*Marigold LCM — resin_front。Affine崩壊（a=73.6）でRMSE=647mm。*

![Marigold resin_oblique](../exp2/depth/marigold_resin_oblique_vis.jpg)
*Marigold LCM — resin_oblique。RMSE=208.2mm。*

| シーン | RMSE | Inlier@2cm | 推論時間 |
|---|---|---|---|
| glass_front | 22.0mm | 97.9% | 4.7秒 |
| glass_oblique | **26.2mm** | 60.9% | 9.7秒 |
| resin_front | 647mm（崩壊） | 1.5% | 9.0秒 |
| resin_oblique | 208.2mm | 10.4% | 4.9秒 |

---

## 7. モデル比較と選定指針

### 7.1 総合比較

#### RMSE とは

**RMSE（Root Mean Square Error：二乗平均平方根誤差）** は、推定値と正解値のずれを表す誤差指標である。本レポートでは「板面の各ピクセルの推定深度が、GT Z（青画用紙で計測した真の距離）から平均的に何 mm ずれているか」を表す。

計算式：

```
RMSE = sqrt( (1/N) × Σ (推定深度ᵢ − GT_Z)² )
```

- N ：板マスク内のピクセル総数  
- 推定深度ᵢ ：ピクセル i の校正後推定深度（mm）  
- GT_Z ：当該シーンの板面 Ground Truth 距離（mm）  

**読み方の目安：**

| RMSE | 感覚的な意味 |
|---|---|
| ～20mm | 2cm 以内のズレ。FoundationPose による把持計画に使用できる可能性がある精度だが、ガラスのような把持マージンが小さいワークでは十分とは言えない場合もある |
| ～50mm | 5cm 以内。おおまかな位置把握には使えるが把持には不足 |
| 100mm 超 | 実用困難。校正崩壊や透過による測定不能が原因の場合が多い |

RMSE はピクセル単位の誤差を二乗してから平均するため、**大きなはずれ値（外れピクセル）の影響が強調される**点が特徴である。少数のピクセルで深度が大きくずれると RMSE が跳ね上がる（樹脂シート正面の崩壊が典型例）。

![RMSE比較](fig_depth_rmse.png)
*各モデルの GT Affine 校正後 RMSE 比較。崩壊シーン（1000mm超）は表示省略。
DA2 がガラス板で最高精度、MoGe-2 が全ワーク対応の唯一モデル。*

#### モデルによって結果が異なる理由の考察

モデル間の差は「透過するかしないか」の二値ではなく、**RGB 上に残る微細な手がかり（エッジ輝線・鏡面反射・スタンド形状）をどれだけ深度推定に活用できるか**によって連続的に変化する。以下に各モデルの差を生んだ要因を考察する。

---

**① ガラス板は「完全に透明」ではない**

そもそもの前提として、実験に用いたガラス板には視覚的な手がかりが複数存在する。
- **エッジ輝線**：板の縁が光を屈折・反射して明るい線として写る
- **鏡面反射**：板面が照明を鏡のように反射する（特に上部）
- **スタンド**：板を立てているスタンドは不透明で確実に写る

これらの手がかりをモデルの深度推定に活かせるかどうかで結果が分かれる。

---

**② MoGe-2：法線推定の同時学習が透明物体検出に効いた可能性**

MoGe-2 は深度に加えて**表面法線マップ（surface normal map）**を同時に推定する点が他モデルと異なる。

**表面法線とは何か**

表面法線とは、ある面が「どの方向を向いているか」を表す単位ベクトルである。
例えば、カメラ正面に立てられた壁の法線はカメラに真っすぐ向かってくる方向（−Z）、床の法線は真上（+Y）を向く。

```
[例]
  壁（正面）  →  法線: カメラ方向（→）
  床           →  法線: 上方向（↑）
  斜め板       →  法線: 斜め（↗）
```

**なぜ法線推定が透明物体の検出に役立つ可能性があるか**

法線と深度の間には幾何学的な整合性制約がある。**深度が急変する場所（エッジ）では法線も急変しなければならない**。
モデルが深度と法線を同時に学習すると、この制約を満たすために両者が互いを補正しながら学習される。

ガラス板のエッジでは、たとえ深度ラベルが「背景と同じ」であっても、**エッジ輝線による輝度の急変は法線の変化として検出できる**。法線推定ネットワークがエッジを「面の向きが変わる場所」として捉えることで、そのフィードバックが深度推定にも波及し、深度マップ上でも板の輪郭が保持されやすくなると考えられる。

また、Fresnel 反射（表面への入射角に依存する反射）はガラスエッジ付近で強くなる。鏡面反射の方向は表面法線と入射光から決まるため、法線推定はガラスの反射パターンを「面の向き情報」として取り込める可能性がある。

**次のステップへの示唆**

MoGe-2 の優位性が法線推定に起因するなら、法線推定を持つ他のモデル（例：Omnidata、DSINE）を透明物体に適用する実験が次の検討候補となる。

---

**③ DA2 Indoor：学習データへの透明物体の混入と近距離特化が効いた可能性**

DA2 の Indoor 特化版は以下の点で透明物体対応に有利な可能性がある。

**学習パイプラインと擬似ラベルの役割**

DA2 は 2 段階の学習を行う。まず小規模な高品質ラベルデータで教師モデルを学習し、その教師モデルを使って大量の**ラベルなし画像に擬似ラベルを付与**して最終モデルを学習する。

Indoor 特化版では室内画像（ガラステーブル・窓ガラス・ガラス扉・水槽など透明物体が日常的に登場する環境）から大量の擬似ラベルを生成している可能性が高い。これらの擬似ラベルは「完璧な正解」ではないが、ガラスエッジ周辺の局所的な深度変化を部分的に捉えていれば、モデルは「透明物体エッジ付近での深度変化パターン」を学習できる。

**近距離特化の利点**

本実験のガラス板は約 380mm という近距離にある。この距離では：
- ガラスのエッジ輝線が画像上で相対的に大きく、視覚的手がかりとして使いやすい
- ガラス面（380mm）と背景壁（650mm+）の深度差が約 70% と大きく、「前にある」という相対的な深度構造が検出しやすい

Indoor モデルは近距離の室内環境に最適化されているため、この条件が活かされた可能性がある。

**Affine-invariant 深度表現の貢献**

DA2 の学習では深度の絶対値ではなく**相対的な深度構造（affine-invariant）**を重視する目的関数を使用する。これにより、「ガラス板が背景より手前にある」という相対的な深度順序を、絶対的な距離値が不正確であっても保持しやすい。

**次のステップへの示唆**

DA2 Indoor の強みが学習データの透明物体混入によるなら、透明物体を意図的に含む fine-tuning データを作成することで他の深度モデルも改善できる可能性がある。また、合成データ（3D レンダリングで透明マテリアルを意図的に含む）を用いた学習実験が有望な方向性となる。

---

**④ Metric3D v2 が特に悪い理由**

Metric3D v2 は「どんなカメラで撮影した画像でも実寸距離を正確に当てる」ことを最優先に設計されており、正準カメラ空間（canonical camera space）への強い正規化処理を行う。この正規化は多様なカメラへの汎化を実現する一方、**エッジ輝線や反射といった局所的な視覚手がかりを均してしまう**方向に働く可能性がある。

結果として、透明領域でも「この外観が示すのは背景の距離である」という判断を確信を持って行い、他モデルが拾えるエッジ付近のわずかな深度変化を出力しない傾向があると考えられる。

---

**まとめ**

| モデル | 透明物体への対応を分けた要因（仮説） |
|---|---|
| MoGe-2 | 法線推定の同時学習によりエッジ・反射を面の向き変化として捕捉できた可能性 |
| DA2 Indoor | 室内近距離データへの特化と affine-invariant 学習が相対深度構造の保持に貢献した可能性 |
| Depth Pro | 境界鮮明化を重視した設計がガラスエッジを捉えた可能性（詳細は未検証） |
| Metric3D v2 | メトリック精度特化の強い正規化が局所的視覚手がかりを削いだ可能性 |

### 7.2 用途別推奨

| 用途 | 推奨モデル | 理由 |
|---|---|---|
| **ガラス板のみ・精度優先** | DA2 | RMSE 19.3mm、Inlier@2cm 98.2%と最高精度 |
| **ガラス板のみ・輪郭品質優先** | Depth Pro | 板の境界が最鮮明 |
| **ガラス板のみ・速度優先** | UniDepth V2 | 130msだが精度は劣る |
| **樹脂シートを含む全ワーク対応** | **MoGe-2** | 唯一全シーン安定。Affine崩壊なし |
| **高品質だが低速でも許容** | Marigold LCM | ガラス板で高精度だが〜10秒 |
| **透明物体には不適** | Metric3D v2 | 透過問題により全シーン精度不足 |

### 7.3 Affine 校正が不要になる条件

2点校正には青画用紙と GT Z 計測が必要で、製造現場ではコストがかかる。
以下の代替手段が検討できる。

| 手段 | 精度 | 準備コスト |
|---|---|---|
| **Affine 校正（本実験）** | ~20mm | 青画用紙 + GT 計測 |
| **ピクセルサイズ法** | ~15mm | カメラ校正 + 既知ワークサイズ |
| **壁校正のみ** | ~50mm | 壁の距離計測のみ |

ピクセルサイズ法は「SAM3 のマスク bounding box 幅 × fx ÷ 実サイズ」で Z を算出する方法で、
本実験での検証では誤差 −14.8mm（約4%）を達成している（詳細は exp2_report.md 参照）。

---

## 8. まとめ

| 項目 | 内容 |
|---|---|
| 最高精度（ガラス板） | DA2：RMSE 19.3mm、Inlier@2cm 98.2% |
| 唯一の全ワーク対応 | MoGe-2：全4シーンで Affine 崩壊なし |
| 透明物体に最も不適 | Metric3D v2：板を透過して背景と同一視 |
| 最大の課題 | 樹脂シート正面での Affine 校正崩壊（MoGe-2 を除く全モデル） |
| 実用推奨パイプライン | SAM3 マスク → MoGe-2 深度 → Affine 校正 → FoundationPose |

---

*次のレポート: [05_depth_stereo.md](05_depth_stereo.md) — Foundation-Stereo によるステレオ深度推定*

---

## 参考文献・リンク

| モデル | 論文 | GitHub |
|---|---|---|
| Depth Anything V2 | [arxiv 2406.09414](https://arxiv.org/abs/2406.09414) | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| MoGe-2 | [arxiv 2507.02546](https://arxiv.org/abs/2507.02546) | [microsoft/MoGe](https://github.com/microsoft/MoGe) |
| Depth Pro | [arxiv 2410.02073](https://arxiv.org/abs/2410.02073) | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) |
| UniDepth V2 | [arxiv 2403.18913](https://arxiv.org/abs/2403.18913) | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |
| Metric3D v2 | [arxiv 2404.15506](https://arxiv.org/abs/2404.15506) | [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) |
| Marigold LCM | [arxiv 2312.02145](https://arxiv.org/abs/2312.02145) | [prs-eth/Marigold](https://github.com/prs-eth/Marigold) |
