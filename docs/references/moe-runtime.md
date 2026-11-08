# MoE 推論の高速化（expert caching / 先読み / 投機）と P40 での適用可否

取得日: 2026-11-07。数値は出典の実測、推定は「推定」と明記する。

## 1. P40 の現状（Strata 実測）

`bench/results/2026-10-12-p40-sm61-kernels/profile-base.txt`（Tesla P40 24 GB, IQ3_S, 8,225 expert slots = 15.6 GiB, RAM に 46.8 GiB, 19 pool workers）:

| 指標 | 実測 |
|---|---|
| expert cache ヒット率（decode） | **99%**（"1-3 ms of CPU experts (hit rate 99%)"） |
| PCIe grp（decode window の PCIe 待ち） | 0.27 ms（GDN）+ 0.09 ms（QSA）= 0.36 ms |
| VRAM hits（VRAM 常駐 expert の計算） | 8.49 ms（GDN）+ 2.67 ms（QSA）= **11.16 ms（22.7%）** |
| waitCPU（CPU expert の計算待ちで GPU が止まる） | 2.50 ms（GDN）+ 0.54 ms（QSA）= **3.04 ms（6.2%）** |
| CPU experts の実時間 | 1.80 ms/window（2.37 entries） |

→ P40 では「expert が VRAM にほぼ全部載っている」ので、**PCIe 先読みの利得は小さい**。効くのは (a) VRAM expert カーネルそのもの、(b) CPU に落ちた 1% の待ち（waitCPU 3 ms）を埋めること。

## 2. 一次情報

### 2.1 HOBBIT: 混合精度 expert offloading

- 出典: Peng Tang et al., "HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference", arXiv:2411.01433 (2024-11-03), https://arxiv.org/abs/2411.01433 （2026-11-07 取得）
- 主張: 「重要度の低い cache-miss expert を低精度版に差し替える」ことで expert-load のレイテンシを大幅に削る。3 つの手法: token-level dynamic expert loading、layer-level adaptive expert prefetching、sequence-level multidimensional expert caching。
- 結果: SOTA の offloading システムに対して **最大 9.93 倍の decode 高速化**（edge device）。
- P40 への示唆: VRAM が足りない時に「miss した expert を小さい量子化で PCIe 越しに送る」選択肢。ただし**出力が変わる**（精度差し替え）ので、Strata の「bit 一致」方針とは相容れない。opt-in の品質トレードとしてのみ検討可。

### 2.2 FreeToken #677: router lookahead で次層 expert を先読み

- 出典: FlashML-org/FreeToken issue #677 "perf(moe): prefetch the next layer's experts from a router lookahead during offloaded decode", 2026-10-13, https://github.com/FlashML-org/FreeToken/issues/677 （2026-11-07 取得）
- 環境: RTX 5080 16 GB, PCIe, Qwen3.8-Flash-Next NVFP4, `--moe-strategy offload`
- 手法: 層 L の router を層 L-1 の router 入力で先に走らせ（残差は隣接層で大きく変わらない）、L-1 の計算中に L の expert をコピーする。予測 expert は「低優先 slot（usage 0, 最初に evict）」に入れる。
- 結果:

| | baseline | prefetch |
|---|---|---|
| decode 32K context | 43.4 tok/s | **53.0 tok/s（+22%）** |
| decode 255K context | 43.0 | 50.8 |
| demand miss / token | 206 | 111 |
| VRAM | — | +148 MiB |

- 正しさ: greedy 出力が 420 の cold nonce prompt で token 一致（slot-map audit 有効）。「lossless」と明記。
- 落とし穴: slot-map の writer を compute stream に一本化しないと 60 prompt 中 1-6 が壊れた（race）。

**Strata への適用可否**: **既に実装済み**。`src/core/expert_source.cpp:3199` の `RouterLookahead::run()` が「層 L+1 の expert を予測して warm する」専用 thread で、confidence / window budget / deadline 付き（`:1127` `warm_scored`）。したがって**再提案しない**。ただし Strata の lookahead は **file tier（experts.bin）専用**で、CPU pool の job 発行には効かない。P40 の waitCPU 3 ms は CPU pool 側なので、同じ考え方を CPU pool に広げる余地はある（→ 候補 F）。

### 2.3 DuoServe-MoE: 二相の expert prefetch と caching

- 出典: "DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for MoE LLM Serving", arXiv:2509.07379, https://arxiv.org/html/2509.07379v2 （2026-11-07 取得）
- 主張: expert 重みを CPU memory へ offload し、prefill 相と decode 相で別の prefetch 戦略を使う。prefetch ベースの offload は burst 的な routing に弱く、cache ベースは expert の再利用性が低いと効かない、という既存手法の弱点を突く。
- P40 への示唆: prefill と decode で expert の再利用パターンが違う（prefill は chunk 内で同じ expert を多数 token が使う）ので、**相ごとにキャッシュ方針を変える**のは理にかなう。Strata は prefill で ring slot を borrow する仕組みが既にある（`--prefill-borrow`, `STRATA_PREFILL_ROUTED_STREAM`）。

### 2.4 FineMoE: latency-memory trade-off

- 出典: Hanfei et al., "Taming Latency-Memory Trade-Off in MoE-Based LLM Serving", EuroSys'26, https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf （2026-11-07 取得）
- 主張: 非活性 expert を CPU に offload する既存研究は、latency と memory のトレードオフを粗く扱う。fine-grained に expert の配置を決める。
- P40 への示唆: 24 GB に 15.6 GiB の expert を置くと残りは約 8 GiB。KV（262K で 3 GiB 級）と prompt buffer を引くと、expert slot を増やす余地は限られる。slot を増やすより「持っている expert を速く読む」方が P40 では効く（後述の R3 実測）。

### 2.5 Strata 自身の expert slot 実測（P100, RESEARCH_RUNS.md）

`docs/RESEARCH_RUNS.md` の R3（P100 16 GB, IQ2_XS, 64K document）:

| expert slots (GiB) | prefill | first token | decode |
|---|---|---|---|
| 2,083 (2.81) | 230 s | 0.69 s | 27.5 tok/s |
| 3,500 | 200.7 s | 0.61 s | 30.9 |
| 5,000 | 200.7 s | 0.55 s | 34.2 |
| 7,000 | 200.7 s | 0.51 s | 36.7 |

- VRAM の限界価値: decode で **0.31-0.41 s / GiB / 200-token answer**（2,000-5,000 slots の間）、5,000 以上では 0.15。
- KV を VRAM に置いても 1 GiB あたりの価値はほぼゼロ（4096/16384/32768/65536 cells で 34.05/34.06/34.21/34.92 tok/s）。
- 「prefill は loan で既に足りている」ので split planner を作っても意味がない。

→ P40 も同じ構造（15.6 GiB の expert + 99% hit）。**slot を増やすより、ヒットした expert の計算を速くする**のが正しい。

## 3. 投機デコード（MoE）

- Strata は MTP draft layer（`src/core/mtp.cpp`）、DFlash2 drafter、prompt lookup を既に持つ。P40 の実測では draft が window の **4.04 ms（7.7%）**、drafts accepted 85/132（64%）。
- 外部の MoE 投機の例: DraftExpert（arXiv 2607.24434, 2026）は expert-offloaded MoE 向けに「expansion-aware self-speculative decoding」を提案。draft を 1 つ学習させ、expert の展開コストを見積もって投機する。
- P40 への示唆: 既に MTP があるので新規導入の余地は小さい。ただし **draft の 4 ms は window の 7.7%** なので、draft の窓幅・accept 率の調整（`--spec`, `--spec-min-p`, `STRATA_SPEC_COST`）は依然としてレバー。`STRATA_SPEC_COST` は sm_121 既定だが arch 非依存なので sm_61 でも試せる（未測定）。

## 4. expert 並列・分散（P40 1 枚では不可）

- Expert parallelism (EP) は複数 GPU 前提。P40 1 枚では使えない。
- 2 枚 P40 の layer split は Strata のコミュニティ実測がある（decode 35.0 tok/s, prompt 350/558 tok/s at 3.9K/21.9K, 2026-10-10）。1 枚より遅いので、この構成は採らない。

## 5. P40 での結論

1. **先読み（router lookahead）は実装済み**。再提案しない。
2. **expert slot を増やす**のは P40 では余地が小さく、限界価値も 0.15-0.4 s/GiB（P100 実測）。優先度低。
3. **効くのは VRAM expert カーネル（11.16 ms, 22.7%）と CPU 待ち（3.04 ms, 6.2%）**。
   - VRAM expert: `STRATA_EXPDEC`（既存 opt-in、2080 Ti で gate/up −11〜−23%）を sm_61 で A/B するのが最短。
   - CPU 待ち: CPU pool の job を 1 層先に発行する（file tier の lookahead を CPU pool に拡張）。これが候補 F。
4. 混合精度 offload（HOBBIT 型）は出力が変わるので、bit 一致方針の外。品質トレードを許すなら選択肢。
