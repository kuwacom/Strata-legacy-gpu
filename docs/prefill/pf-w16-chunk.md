# PF_W16 cache と prefill chunk の A-B-A（v3 / 512K / P40）

作業ツリー: `~/work/wt/pfw`（branch `docs/pfw16`、ソース変更なし）
イメージ: `strata:fork-v3`（engine 0.1.57-linux-x86_64-cuda-cuda12）
本番相当の起動設定: `STRATA_SET="ple-io=ram prefill=auto STRATA_DENSE_F32=1 rope-scaling=yarn rope-scale=2 STRATA_DENSE_MMQ=1 STRATA_MMQ_KQ=1 spec=4"` + `--max-context 524288 --kv int8 --kv-resident 32768 --spec 4 --draft dflash2`
生ログ: `~/work/results/pfw/`（アーム別に `<label>-bench.jsonl` / `<label>-keylines.txt` / `<label>/engine-cmdline.txt` ほか）

## 計測条件

- `BENCH_SEED=42`、`max_tokens=1`、TTFT は最初の content/reasoning トークンまで、
  prefill tok/s = usage.prompt_tokens / TTFT（`prefillBench.py`）
- A-B-A: A1（ベース）-> B1（候補）-> A2（ベース再）。各アーム 24K prose/code x3、
  128K prose/code x3（A2 は 24K x3 x2 + 128K prose x2）。1 本目除外はしないで全 run を
  記録し中央値で比較（雛形 `runV3b.sh` と同じ扱い）。判定は 5% 閾値
- 全アームに `STRATA_LOGITS_DUMP` を付け、各アームの最初のリクエストを同一 24K プロンプトの
  KL 用リクエスト（`klreq.py`、rec 0）にした。固定 4 プロンプトの回答一致は `answers4.py`
- 全アームに `STRATA_ARENA_RAM_WAIT_S=120`（前エンジン退出待ちで 20 秒で蹴られた
  pknobs の W2 失敗の回避。起動待ちのみで計測値に無関係）
- ロックは別タスク（pknobs-v3b-p1/p2）の合間に取得。各ブロック 20-38 分、
  ブロック終了ごとに本番を `prod.env` から復元してロック解放

## 1. STRATA_PF_W16_CACHE_MIB=auto

| arm | code-24k | code-128k | prose-24k | prose-128k |
|---|---|---|---|---|
| A1 | 594.3 | 589.4 | 587.1 | 586.8 |
| B1 (W16) | 594.3 | 589.7 | 589.5 | 588.2 |
| A2 | 594.6 | - | 592.4 | 588.5 |

- 差: B1 vs A1 で code-24k +0.00% / code-128k +0.05% / prose-24k +0.41% / prose-128k +0.24%。
  一方 A2 vs A1（ノイズフロア）は code-24k +0.05% / prose-24k +0.90% / prose-128k +0.29% で、
  B1 の差はすべてノイズフロア以下。**差なし**
- VRAM / expert slot: gpu-end は両アームとも 23861 MiB、slot も同一
  （9801 -> 10829 への再 sizing も同じ）。FP16 weight cache の割り当てログなし
- 構造上の理由（ソース裏付け）: `src/prefill/gemm.cu:1405-1462` と
  `include/strata/prefill/gemm.hpp:106`。`=auto` は重みをキャッシュする条件が
  「cudaMemGetInfo の空き > 重みサイズ + kW16AutoReserve(3 GiB)」。P40 512K では
  expert cache（15.9-17.6 GiB）+ prompt borrow（3.99 GiB）が VRAM を満たし、
  空きが 3 GiB を超えないため**何もキャッシュされない**。ログのコメント通り
  「a card whose expert cache fills it caches nothing」
- logits: A1 vs B1 rec 0（同一プロンプト、prefill 直後の窓）は **bitwise 一致**。
  レコード全体では median KL 4.4e-4 / bitwise 10/73 だが、A1 vs A2（どちらもベースの
  クロス再起動比較）でも median KL 2.1e-6 / bitwise 14/30 / KL>0.1 が 2 レコードあり、
  spec 4 の decode 窓の揺れはノイズフロアと同じ。回答の固定 4 プロンプトも
  id 2 / id 4 が A1-A2 間でも同程度に変わる（id 1 / id 3 / needle は全アーム一致）
- 判定: **差なし（P40 512K では構造的に no-op）**。採用しても速くならない。
  `--batch` を使う構成や expert cache を縮小する構成では初めて効く（ソースの注記）。
  採用時の STRATA_SET 追加: なし（現状維持を推奨）
- リスク: なし（何も割り当てられないため）。むしろ `=auto` を入れておくと
  expert cache を縮小した構成で突然 VRAM を取られる点には注意

## 2. prefill chunk 4096（`--prefill 4096`）

| arm | code-24k | code-128k | prose-24k | prose-128k |
|---|---|---|---|---|
| A1 | 594.3 | 589.4 | 587.1 | 586.8 |
| D4096 | 559.7 | 548.5 | 558.0 | 549.0 |

- 差: **-4.96% / -6.94% / -5.82% / -6.45%**。TTFT も 24K +2.6s / 128K +10.4s。
  5% 閾値を超える悪化。エンジンログの自己計測でも 25723 トークン 559.4 tok/s と整合
- 品質: rec 0 は A1 と **bitwise 一致**（8192 以下の chunk は CUDA exact、docs 通り）。
  回答は `~/work/results/pfw/D4096-answers.jsonl`（A1 と同一の傾向）
- VRAM / slot: expert cache は A1 と同じ。prompt borrow も起動時 2760 slots（4.51 GiB）で同一
- 判定: **悪化**。採用しない。`STRATA_SET` は `prefill=auto`（8192）のまま

## 3. prefill chunk 16384（`--prefill 16384`）

| arm | code-24k | code-128k | prose-24k | prose-128k |
|---|---|---|---|---|
| A1 | 594.3 | 589.4 | 587.1 | 586.8 |
| D16384 | 611.4 | 609.6 | 608.9 | 609.9 |

- 差: **+2.88% / +3.43% / +3.71% / +3.94%**（TTFT 24K -1.1s / 128K -14.2s）。
  4 セルすべてで同方向の一貫した加速だが、5% 閾値は下回る。**条件付き**
  （ノイズフロア +0.9% に対して一貫 +3% 台なので実質的には有効と推定するが、
  別日での再現は未確認）
- 品質: rec 0 は **not bitwise**（docs の 8192 超は not bit-exact 通り、実測で確認）。
  回答の固定 4 プロンプト: id 1 / 2 / 3（needle QV-4821-ZT 含む）は A1 と同一、
  id 4 は言い換え（A1-A2 のノイズフロアでも id 4 は変わるので品質低下の証拠にはならない）。
  KL 定量は未実施（回答一致のみ）
- **副次効果（重要）**: chunk 16384 では prompt path の borrow が 2760 slots（4.51 GiB）
  -> 4633 slots（7.52 GiB）に増え、これを見た Terminal の drafter が `--draft-conf` を
  0.7 -> 0.8 に変えた（`strata_control/drafting.py:231` の CONF_STEPS、
  推定 slot 数で段階選択）。expert cache 本体は 10311 slots で A1 と同水準、
  decode expert hit rate も 90.1% vs 89.8% で影響は小さい
- 判定: **条件付き採用**。prefill だけなら +3% 台で有効。ただし採用するなら
  `--draft-conf 0.7` を明示して CONF_STEPS の連動を止めるか、0.8 のままの
  品質/速度影響を別途確認する必要がある（今回は draft-conf の A/B はしていない）
- 採用時の STRATA_SET 例: `prefill=auto` -> `prefill=16384`（`--draft-conf 0.7` の
  明示を推奨。未確認項目は下記）

## 4. STRATA_SELECT_SIMT=1

（simt ブロックの結果は `qsa-simt.md` に記載）

## 判定一覧

| 候補 | prefill 差 | 品質 | VRAM/slot | 判定 |
|---|---|---|---|---|
| `STRATA_PF_W16_CACHE_MIB=auto` | 差なし（ノイズ以下） | rec 0 bitwise、ノイズフロアと同程度 | 影響なし | **差なし（no-op）** |
| chunk 4096 | -5.0〜-6.9% | rec 0 bitwise | 影響なし | **悪化** |
| chunk 16384 | +2.9〜+3.9% | rec 0 not bitwise、回答は許容範囲 | borrow 4.51->7.52 GiB、draft-conf 0.7->0.8 に連動 | **条件付き** |

## リスク

- chunk 16384 は not bit-exact。長文の agent ループで出力が変わることを許容できるかはオーナー判断
- `--draft-conf` の連動は暗黙に decode 挙動を変える。採用时は明示指定を推奨
- chunk 16384 の 480K / 長時間会話での挙動、KL 定量は未計測
- A-B-A は別ブロックに分割したため、A2 は A1 の約 90 分後の計測。日内温度ドリフト（69-73C）は
  clocks 1531 MHz 固定で吸収されているとみなす

## 未確認・逸脱

- 128K の run 数: A1/B1 は code/prose 各 3、D4096/D16384 は各 2（指定通り）。
  50 分ブロック制約のため A2 の 128K は prose x2 のみ
- W2（pknobs の未完了分）は本ブロックで B1 として完了。pknobs 側のディレクトリには書いていない
- `STRATA_ARENA_RAM_WAIT_S=120` を全アームに追加（雛形からの逸脱。起動待ちのみ）
- logits dump により各アーム +455-485 MB のボリューム一時ファイル。回収後に volume からは削除済み
- chunk 16384 の KL 定量、`--draft-conf 0.7` 固定での 16384 再計測は未実施
