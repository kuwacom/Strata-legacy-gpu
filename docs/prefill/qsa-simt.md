# QSA select の SIMT scorer（STRATA_SELECT_SIMT=1）の A/B（v3 / 512K / P40）

作業ツリー: `~/work/wt/pfw`（branch `docs/pfw16`、ソース変更なし）
イメージ: `strata:fork-v3`（engine 0.1.57）、本番相当設定は `pf-w16-chunk.md` の計測条件と同一
生ログ: `~/work/results/pfw/`（S1 = 128K ベンチ + KL、S2 = needle + 480K）

## 何を測ったか

`STRATA_SELECT_SIMT=1` は QSA select の score 計算を TF32 テンソルコアではなく
FP32 tiled SIMT カーネルで行う opt-in（`src/kernels/cuda/qsa_select.cu:2215-2295`）。
summation order が違い、設計上非 bitwise。P40（sm_61）は select をテンソルコアで
3xTF32 で走けている（`src/prefill/prefill.cpp:4293` のコメント）ので、これを FP32 の
SIMT に置き替えたときの prefill 速度と品質を確認した。

## prefill 速度

| arm | prose-128k | code-128k | 480K (1 run) |
|---|---|---|---|
| A1（ベース） | 586.8 (3runs) | 589.4 (3runs) | - |
| S1 (SIMT) | **595.4** (3runs) | **596.1** (3runs) | - |
| S2 (SIMT) | - | - | **543.7** (TTFT 958.2s, 520,982 tokens) |

- 128K: **prose +1.47% / code +1.14%**。24K は未計測（指定は 128K のみ）
- 480K: タスク指定の基準値「v3/512K で 480K 約 529 tok/s」と比べ **+2.8%**。
  ただし別日の別プロンプト構成との比較なので**参考値**（A/B ではない）
- 判定: 5% 閾値は下回るが、A-B-A のノイズフロア（最大 +0.90%）
  を上回る一貫した加速。128K/480K の両方で同方向。**条件付き**
  （効果は小さい。採用しても P40 の prefill を 1-3% 速める程度と推定）

## 品質

- KL（`klreq.py` 24K、両アームの最初のリクエスト rec 0 を突き合わせ）:
  S1 は要求種を 128K のみにしたため A1 とレコード番号が揃うのは rec 0-11 のみ。
  その範囲で **median KL 0 / 10-12 レコード bitwise / KL>0.01 が 1 レコード /
  KL>0.1 は 0**。rec 0（24K prefill 直後の窓）は bitwise 一致
  - 注意: 揃う窓が 12 レコードと短く、128K の長い prefill 全体の KL 分布は
    見えていない。これは測定設計の制限（未確認を明記）
- needle（`needle.py`、pos 50%、needle PFW-7734-KM、max 96 tokens）:
  - 32K: **found=true**（prefill 609.3 tok/s、答えは完全一致）
  - 128K: **found=true**（prefill 596.8 tok/s）
- 固定 4 プロンプトの回答一致（answers4）は SIMT アームでは実施していない。
  回答レベルの差は KL と needle のみで確認

## VRAM / expert slot

- S2 の expert cache は 10311 slots / 16.72 GiB でベースアームと同水準、
  prompt borrow も 2760 slots（4.51 GiB）で同じ。SIMT は VRAM を消費しない

## 判定

**条件付き採用**。

- 採用時の STRATA_SET 追加: `STRATA_SELECT_SIMT=1`（env 追加のみ、起動フラグ変更なし）
- 根拠: 128K +1.1〜1.5%（A-B-A、ノイズフロア以上）、480K 参考 +2.8%、
  needle 32K/128K とも成功、揃った窓の KL は実質 0、VRAM/slot 影響なし
- 留保: 効果が 5% 閾値未満。非 bitwise は設計上明記されているものの、
  今回の揃った範囲では差がほぼ出ていない。別プロンプトで routing が変わる
  可能性は残る（未確認）

## リスク

- summation order が違うため、プロンプト次第で expert routing が変わる理論上の可能性。
  長文 agent ループで出力が揺れる許容はオーナー判断
- 24K での速度・品質は未計測。KL も 12 レコード分しか揃っていない
- 480K の +2.8% は基準値との参考比較（A/B ではない）で、誤差込みの可能性

## 生ログ

- `~/work/results/pfw/S1-bench.jsonl` / `S1-kl.json` / `S1-logits.bin`
- `~/work/results/pfw/S2-needle.jsonl` / `S2-bench.jsonl` / `S2-gpu-end.txt`
- KL 解析: `python3 ~/work/results/v3/scripts/kl_by_rec.py ~/work/results/pfw/A1-logits.bin ~/work/results/pfw/S1-logits.bin`
