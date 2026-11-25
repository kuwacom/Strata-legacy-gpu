# perf/ink-final2: S_K dn の shared 版を既定化 + 512K 通し測定

測定日: 2026-11-25 / 対象: `~/work/ink-ap` branch `perf/ink-final2` (`b135e756a`、base `perf/ink-dn-shared` `f417d78ee`) / バイナリ sha256 `941986846e52…` / Tesla P40 (sm_61) 単機 / SQ-10。本番は停止 -> 復旧済み (healthy、`.env` sha256 `(省略)` 不変、GPU/ビルド両ロック解放)

## 結論

- **完走**。512K (k8v4、chunk 512、draft off、gen 256) で **prefill 114.2 ppts / 4,592.52 s**。前回 (dspark、gen 256) の **111.8 ppts / 4,691.47 s** に対して **+2.4 ppts (+2.15%) / −98.95 s** で合格条件 (前回 111.8 超え) を満たす
- **dn(sk) 310.43 -> 197.34 ms/step (−113.09、−36.4%)**。gu(sk) は 569.25 -> 568.40 (−0.15%) で、32K / 128K と同じく dn だけで説明できる
- **host スピルなし**: `experts=3072 VRAM / 0 host (hyb=0, promo=0, pf_buf=0MB)`、実行中の `Host KV` / `spill` / `offload` 行は 0 件 (`--kv-offload off`)
- **DSpark の差**: 同じ gen 256 で **dspark 5.2 tps (49.09 s) に対し draft off 7.7 tps (33.21 s)**。512K では dspark の方が 2.5 tps (32%) 遅い、という差が出た (今回は draft off 1 本だけなので断定しない)
- VRAM ピークは **17,499 MiB** (前回 20,037 MiB)。ただし dspark の有無 (`screen=1 -> 0`、DSpark 78 -> 0 MiB) とバイナリが違うため単純比較はできない

## 変更とビルド

| 項目 | 内容 |
|---|---|
| 変更 | `ink_sk_dn_smem_mode()` の既定を 0 -> **1** (`STRATA_INK_SK_DN_SMEM=0` で従来の global 直読に戻る)。コメントと実測値を追記。1 ファイル +5 / −3 行 |
| commit | `b135e756a` `perf: S_K dn の shared 版を既定化` (author `kuwacom <kuwacom@users.noreply.github.com>`、push なし) |
| バイナリ | 6,943,136 B、sha256 `941986846e52b39058a913abf27732e1e302170e475623a2f49ed3141601aa8a` |
| cubin | `strata-inkling.1.sm_61.cubin` / `strata-inkling.2.sm_61.cubin` の 2 本のみ (PTX なし)。`~/work/results/ink-final2/cubins.txt` |
| ビルド | ビルドロック、JOBS=4、`--memory=10g --memory-swap=10g`、`CMAKE_CUDA_ARCHITECTURES=61-real` (227 s) |

## 512K 比較表

条件: `--max-context 524288 --parallel 1 --chunk 512 --kv k8v4 --kv-offload off --prompt-len 524288 --gen 256`、`STRATA_INK_PF_TILE=1 STRATA_INK_PROF=2`、`STRATA_INK_SK_DN_SMEM` は今回未指定 (= 新既定 1)。

| 項目 | 前回 (`ink-final`、dspark) | **今回 (`ink-final2`、draft off)** | 差 |
|---|---|---|---|
| prefill | 111.8 ppts | **114.2 ppts** | **+2.4 ppts (+2.15%)** |
| prefill 所要 | 4,691.47 s | **4,592.52 s** | **−98.95 s** |
| dn(sk) ms/step | 310.43 | **197.34** | **−113.09 (−36.4%)** |
| gu(sk) ms/step | 569.25 | 568.40 | −0.85 (−0.15%) |
| attn 合計 (1024 step の平均 x 1024) | 2,906.5 s (2,838.35 ms/step) | 2,927.7 s (2,859.03 ms/step) | +21.2 s (+0.7%) |
| decode 全体平均 (gen 256) | 5.2 tps (49.09 s) | **7.7 tps (33.21 s)** | **+2.5 tps (+48%) / −15.88 s** |
| decode 1 token あたり | 191.8 ms | 129.7 ms | −62.1 ms |
| decode 後半 128 token | 未取得 | 未取得 | — |
| VRAM ピーク | 20,037 MiB | **17,499 MiB** | −2,538 MiB |
| host スピル | なし | **なし** | — |
| 完走 | あり | **あり (exit 0、PROF 1024 blocks)** | — |

- prefill の差 −98.95 s は、dn の短縮 −113.09 ms/step x 1024 = −115.8 s と attn の増加 +21.2 s の合計にほぼ一致する。attention と SF4/router は同じカーネルで、dn 以外に説明できる変化は無い
- attn 合計の +21.2 s (+0.7%) は 32K / 128K の A/B (attn ±0.1%) より大きいが、前回と今回でバイナリが違い (`ink-final` -> `ink-final2`)、512K は各 1 本なので**ばらつきかどうかは未確認**。今回の合格判定には影響しない (それでも prefill は向上)
- decode の「後半 128 token」は `--bench` が合計しか出さないため分離できない (per-token のログが無い)。両 run とも全体平均のみ
- 終了時 clock 1531 MHz / 73 C / 166.9 W / GPU 使用率 99% で throttle なし。VRAM ピークは `vram.csv` (5 s 間隔) の最大値 `17,499 / 6,941 MiB`

## DSpark の差 (断定しない)

- 前回は `--draft dspark`、今回は `--draft off`。同じ gen 256 での decode は **5.2 -> 7.7 tps (+2.5 tps、+48%)** で、**512K では dspark の方が遅い**という差になった
- dspark は 1 step で複数 token を出すが、512K では 1 step のコスト自体が大きく、accept が伸びないと損になる、という説明が可能。ただし
  - dspark 側は前回 1 本だけで、今回のバイナリとも別 (`ink-final` vs `ink-final2`)
  - decode は M=1 で dn の smem 版 (MAX_M=16 専用) を通らないため、今回の変更そのものは decode に効いていない
  - したがって「512K では dspark が不利」とまでは言えず、**この条件の差**として記録する
- VRAM も dspark 側が 2,538 MiB 多い (78 MiB の DSpark バッファだけでは説明できない。`screen=1` の有無が関係する可能性があるが未確認)

## 逸脱・注意

- 前回値は `~/work/results/ink-final/B-512k.*` (dspark、gen 256) を使った。タスク指示の 111.8 ppts / 4,691 s / 5.2 tps / 20,037 MiB / attn 2,906.5 s と一致することを確認済み (attn は 1024 block の平均 x 1024 で再計算)
- 今回は env 未指定で回し、新既定 (1) が効いていることをログで確認した (`dn(sk)` が 197 前後、`DSpark took 0 MiB`)
- decode 後半 128 token の平均は取得できていない (`--bench` の出力は合計のみ)。数値を作っていない
- VRAM ピークの差は dspark/screen とバイナリ差の交絡があり、原因は未確認
- attn 合計 +0.7% の原因も未確認 (単発アーム)
- 512K の所要は 4,592.52 s (prefill) + 33.21 s (decode) + モデルロード約 20 s。ロック保持は 18:03:53 -> 19:25:00 (復旧完了)

## 次の提案

1. **`perf/ink-final2` を本線候補にする**。32K +6.0% / 128K +4.6% / 512K +2.15%、数値は bit 一致
2. **512K の dspark on/off を同一バイナリで 1 本ずつ**取り、decode の差 (5.2 vs 7.7) が構成差かバイナリ差かを切り分ける。decode だけなら 1 時間弱で回せる
3. attn (+0.7%) の再現性を確認するため、512K をもう 1 本 (計 2 本) 回してばらつきを見る
4. 512K の prefill はまだ attn が 64% (2,927.7 / 4,592.5 s)。次の大枠は attention 側で、S_K dn はここで打ち止めに近い
