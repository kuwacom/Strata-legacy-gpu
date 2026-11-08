# PATTN 検証（strata:fork-v3 / engine 0.1.57 / 512K / spec=4 / dflash2）

## 目的

`STRATA_PATTN_SM60=1`（prefill 24K/128K で +6.5〜6.6% と先行計測された）を、本番相当設定
（512K、spec=4、DENSE_MMQ/MMQ_KQ、dflash2、conf 0.7）で再検証し採用可否を確定する。

先行データ: `~/work/results/pknobs/v3/`（C1=PATTN off、C2=on、C3=off）。
本検証の生ログ: `~/work/results/pknobs/v3b/`（ブロック p1〜p4）。

## 条件

- ベンチ: `prefillBench.py`、`BENCH_SEED=42`、1 本目除外、中央値、判定は 5% 閾値。TTFT 併記
- A = PATTN off（A1/A2）、B = `STRATA_PATTN_SM60=1`。A-B-A で実施
- アーム起動は pristine state/config のコピー毎（preset 汚染なし）、`--cpuset-cpus 36-47`
- モデルは IQ3_XXS を明示（`-e MODEL=IQ3_XXS`、`STRATA_SET` は本番 prod.env と同一）

## 速度（新規実測 v3b）

| 条件 | A1 (off) | A2 (off) | B (on) | B vs A1 | A 揺れ (A1 vs A2) |
|---|---|---|---|---|---|
| 24K prose | 589.5 tps | 585.0 | 629.8 | **+6.8%** | -0.8% |
| 24K code | 590.5 tps | 590.3 | 632.2 | **+7.1%** | -0.0% |
| 128K prose | 586.5 tps | 584.6 | 625.0 | **+6.6%** | -0.3% |
| 128K code | - | 590.6 | - | - | - |

TTFT 中央値: 24K 42.2s -> 39.5s、128K 229.3s -> 215.1s（B が速い）。

先行計測（C1/C3 off、C2 on）との整合:

| 条件 | C1/C3 (off) | C2 (on) | 先行差 |
|---|---|---|---|
| 24K prose | 587.4 / 591.1 | 625.3 | +6.4% |
| 24K code | 594.1 / 590.9 | 630.2 | +6.1% |
| 128K prose | 588.6 / 588.95 | 627.1 | +6.5% |

- 差は新規・先行とも +6.1〜+7.1% で一貫。A 同士の揺れ（-0.8〜+0.8%）の 8 倍以上
- 5% 閾値は 24K/128K ともにクリア

## 品質

### needle 128K（50% / 90%、retrieval、max 24 token）

| | 50% | 90% |
|---|---|---|
| A1 (off) | found | found |
| B (on) | found | found |

### 128K 長文 QA 8 問（memo 埋め込み）

| | スコア |
|---|---|
| A1 (off) | 8/8 |
| B (on) | 8/8 |
| A2 (off) | 8/8 |

### 固定 4 プロンプト（24K、temp 0、max 128）

`ans4-<arm>-pass*.json` の diff 行数。

| プロンプト | A vs B | A 自己 (pass1 vs pass2) |
|---|---|---|
| 1 見出し列挙 | 一致 | 一致 |
| 2 KV キャッシュ要約 | 4 行差 | 4 行差 |
| 3 埋め込みコード抽出 | 一致 | 一致 |
| 4 parse_md_tables コード | 一致 | 17 行差 |

- プロンプト 2 の差は 128 token 打ち切り位置の言い換えで、内容は同じ主張の継続
- プロンプト 4（コード生成）は A 自己で `List[Dict]` vs `list[dict]` と docstring 言い換えの揺れがある一方、
  A vs B はバイト等一致。B の差分は A 自己自己揺れの範囲内で、意味的に同じ
- 先行計測（C1 を 2 回、C2）でも #2/#4 に同程度の揺れがあり、PATTN 固有の変化とは言えない
- 誤答・事実誤認・抽出漏れは A/B ともに見つかっていない（needle・QA・#3 の抽出は全一致）

### 480K needle（50%、1 回、ブロック p4）

| | 結果 |
|---|---|
| B (on) | **found**（prompt 470,287 tok、prefill 561.6 tps、TTFT 837.4s、エラーなし） |

A (off) での 480K は未実施（時間制約）。480K での prefill 速度比較は未確認。

## 判定

- **速度: 採用条件を満たす**。prefill 24K/128K とも +6.6〜+7.1%（先行 +6.1〜+6.5% と一貫）、
  A 自己揺れ -0.8〜+0.8% の 8 倍超、5% 閾値クリア。TTFT も同率で改善
- **品質: 劣化なし**。needle 128K 50/90%・480K 50% すべて発見、128K QA 8/8（A も 8/8）、
  固定 4 プロンプトの差分は A 自己自己揺れの範囲内で意味的に同じ（コード生成は A vs B 完全一致）
- **リスク: 低**。出力が揺れるのは PATTN 固有ではなく temp 0 でも起きる既知の非決定性
  （先行 C1 の 2 回計測でも同様）。ただし 480K needle は 1 回のみで統計は薄い
- **採用可否: 採用**
- 採用時の STRATA_SET（本番 prod.env に追記する 1 トークン）:

```
STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4 STRATA_PATTN_SM60=1"
```

  （`-e STRATA_PATTN_SM60=1` の直接 env でも同じ。検証時は STRATA_SET 経由でも直接 env でも
  engine 起動ログで pattn 有効を確認すること）

## 逸脱・注意

- 128K bench は prose のみ 3 回（code は 24K のみ。先行 C1-C3 も 128K は prose のみ）。
  128K code の A 側 2 回を A2 ブロックで補完（B 側は未実施。24K code が +7.1% なので影響小と判断）
- needle 480K は B のみ 1 回。A との 480K 比較は未確認
- ブロック間で他タスク（pfw-*）が GPU ロックを奪い合ったため、p3 は 3 回リトライ、
  p4（480K のみ）を別ブロックに分離した。その間の本番停止は各タスクが復元済み
- prod-restore-*.txt がブロック終了数分後に消える現象が 2 回発生（内容は自ブロックのログに
  "prod health: healthy" として残る。削除元は不明・他タスクの活動と時間が一致）

