# 512K context の実利用品質（YaRN scale 2 / int8 KV）

## 結論

- 本番の 512K 構成（`--max-context 524288`、`--kv int8`、`--kv-resident 32768`、rope yarn factor 2）は、合成長文の needle retrieval を **480K トークンのプロンプトまで全件正答**した。プロンプト長 33K / 130K / 260K / 480K のいずれでも、埋めた ID を 14 トークンでそのまま返した
- 480K の TTFT は 1,533 秒（25.6 分）、prefill は 302.6 tok/s。OOM・エラー・stall は出ていない。VRAM 空きは 289-310 MiB で推移し、RAM 使用は最大 87.1 GB、RAM 利用可能は最小 13.4 GB
- decode は 480K の後でも 42.6 tok/s（steady、14 トークンのバースト計測）。#2065 警告が言う「数 tok/s まで落ちる」状態にはならなかった
- **128K 以内と 128K 超過で retrieval 精度の差は出なかった**（両側とも全件正答）。ただし needle の絶対位置は最大 240K トークンで、学習コンテキスト 262,144 を超える位置には置いていない。YaRN の外挿域（262K 超の位置）の recall は未検証
- prefill 速度はプロンプト長に対して単調に低下した（33K 474 tok/s -> 129K 418 -> 260K 366 -> 480K 303）。深いほど遅い
- 60 分ロックに収まらず、位置グリッドは 12 セル中 7 セルで打ち切った。32K@10 / 128K@10 / 256K@10 / 500K@90 / 500K@10 / 480K@90 は未実施

## 検証条件

- 本番 `strata`（image 0.2.7-cuda12、engine 0.1.52）を再起動せず、HTTP `localhost:8080/v1/chat/completions` のみで計測した
- コンテナ起動時刻は 2026-11-07T05:59:40Z。計測中に再起動はしていない（終了確認でも `StartedAt` は不変）
- 合成プロンプトは `~/Strata/docs/*.md` を連結・繰り返して作った。文字数 = 目標トークン数 × 3.25 で切り、needle 文を文字位置の 10/50/90% に 1 行挿入した
  - needle 文: `The access code for this document is <ID>. Remember it.`
  - 末尾の質問: `What is the access code mentioned in the document above? Reply with only the code.`
- 各セルで別の ID を使った（prefix 再利用時に前の会話の ID を拾っていないかを見るため）。ID は `NEEDLE-XXXX-YYYY` 形式
- temperature 0、`enable_thinking: false`、max_tokens 64、stream + `include_usage`
- 実際の `prompt_tokens`（usage）は目標より小さい。文字/トークン比の実測は 3.22-3.33 だった

| 目標 | 実測 prompt_tokens | 文字数 | 文字/token |
| ---: | ---: | ---: | ---: |
| 492,000 | 479,998 | 1,599,170 | 3.33 |
| 262,144 | 260,172 | 852,138 | 3.28 |
| 131,072 | 129,699 | 426,154 | 3.29 |
| 32,768 | 33,114 | 106,666 | 3.22 |

- GPU ロックは 06:17:20 に取得し 07:20:53 に解放した（63.5 分。60 分を 3.5 分超過。理由は後述）
- VRAM/RAM は 5 秒間隔の `nvidia-smi` と `free -m` で記録した（`~/work/results/quality/monitor-seq.log`）

## retrieval 精度（needle in a haystack）

| プロンプト長 | needle 位置 | 実測 prompt_tokens | needle 絶対位置（推定） | 結果 | 根拠 |
| ---: | ---: | ---: | ---: | --- | --- |
| 33K | 50% | 33,114 | 16,557 | 成功 | 応答 `NEEDLE-4D9C-1064`（14 token） |
| 33K | 90% | 33,114 | 29,803 | 成功 | 応答 `NEEDLE-8B2E-9953`（14 token） |
| 33K | 10% | 33,114 | 3,311 | 未実施 | ロック時間切れ |
| 130K | 50% | 129,699 | 64,850 | 未確認 | サーバ側は 14 token 生成で完了（`prompt 129699 tokens = 65536 reused + 64163 read ... 14 generated`）。クライアントを kill したため応答本文を取得できず、ID の一致は未確認 |
| 130K | 90% | 129,699 | 116,729 | 成功 | 応答 `NEEDLE-3E7B-5590`（14 token） |
| 130K | 10% | 129,699 | 12,970 | 未実施 | ロック時間切れ |
| 260K | 50% | 260,172 | 130,086 | 成功 | 応答 `NEEDLE-5D2C-4477`（14 token） |
| 260K | 90% | 260,172 | 234,155 | 成功 | 応答 `NEEDLE-2B6E-0311`（14 token） |
| 260K | 10% | 260,172 | 26,017 | 未実施 | ロック時間切れ |
| 480K | 50% | 479,998 | 239,999 | 成功 | 応答 `NEEDLE-7F3A-1928`（14 token） |
| 480K | 90% | 479,998 | 431,998 | 未実施 | ロック時間切れ |
| 480K | 10% | 479,998 | 47,999 | 未実施 | ロック時間切れ |

- 8K の予備計測（8,172 tokens、位置 50%）でも正答した（`CAL-0000`）。短い側の対照として使える
- 成功はすべて「ID だけを 14 トークンで返した」形。ID は 16 文字だがトークナイザで 14 トークンになる
- 失敗は 1 件も観測していない。ただし **試行回数が少ない**（各セル 1 回）。確率の推定には使えない
- 位置 10% のセルはすべて未実施なので、先頭付近の recall は測れていない

## 500K プロンプトの TTFT・prefill・メモリ

- 目標 492,000 トークンで送り、実測 479,998 トークン（`--max-context 524288` に対して 44,290 の余裕）
- TTFT 1,533.4 秒、engine の prefill は 302.6 tok/s（463,614 トークンを 1,532,163 ms）
- 14 トークン生成に 412 ms（34.0 tok/s）
- OOM なし。`cudaMalloc` 失敗、stall、`no progress` はいずれも計測区間に出ていない（stall 行は engine 起動直後の 06:00 台のもので、今回の区間ではない）
- 会話キャッシュへの parking は見送られた（`skip parking (snapshot 7786 MiB exceeds available budget)`）。480K の会話は 7,168 MiB の枠に収まらない

| プロンプト長 | VRAM 使用 最大 | VRAM 空き 最小 | RAM 使用 最大 | RAM 利用可能 最小 |
| ---: | ---: | ---: | ---: | ---: |
| 480K | 24,131 MiB | 309 MiB | 87,127 MB | 13,442 MB |
| 260K | 24,133 MiB | 307 MiB | 87,080 MB | 13,489 MB |
| 130K | 24,133 MiB | 307 MiB | 85,466 MB | 15,103 MB |
| 33K | 24,151 MiB | 289 MiB | 86,554 MB | 14,016 MB |

- VRAM は idle の 24,131 MiB からほぼ動かない。KV は int8 で 6.19 GiB を pinned RAM に置き、VRAM は 32,768 cells に固定しているため
- RAM は計測前の idle 82.2 GB から最大 87.1 GB まで増えた。page cache を追い出しながら pinned arena を維持している状態と見られる（推定）
- 500K 開始時に engine が `strata verify: capturing the 3-token window (306 MiB of VRAM free)` を出した。空き 306 MiB でも 3-token の verify は通っている
- #2065 警告は計測開始直後の短いリクエストで 2 回出た（494 MiB / 310 MiB）。500K 以降の長文リクエストでは新しい警告は出ていない。VRAM 空きは 289-310 MiB のままなので、抑止条件は未確認

## 長文 decode

- 長文の後で 128 トークンを生成する計測は時間切れで実施できなかった。代わりに retrieval 応答の 14 トークンバーストから decode 速度を出した（参考値）
- 14 トークンでは最初の数トークンに warmup が混ざるため、engine の生成時間と、チャンク時刻の後半 3/4 から出した steady 値を併記する

| プロンプト長 | 生成トークン | 生成時間 | decode tok/s（全体） | decode tok/s（steady） |
| ---: | ---: | ---: | ---: | ---: |
| 480K | 14 | 412 ms | 34.0 | 42.6 |
| 260K（90%） | 14 | 297 ms | 47.2 | 48.3 |
| 260K（50%） | 14 | 370 ms | 37.8 | 34.6 |
| 130K（90%） | 14 | 275 ms | 51.0 | 54.8 |
| 130K（50%） | 14 | 298 ms | 47.0 | 未算出 |
| 33K（90%） | 14 | 264 ms | 53.1 | 56.6 |
| 33K（50%） | 14 | 262 ms | 53.5 | 57.2 |

- 480K でも decode は 40 tok/s 前後を維持した。128K 本番の基準（45-50 tok/s）から 1-2 割落ちる程度
- 14 トークン計測なので ±10% 程度のばらつきはありうる。128 トークンでの再計測が望ましい

## 128K 以内と超過の比較

- 128K 以内（prompt <= 131,072）: 33K@50 成功、33K@90 成功、130K@90 成功、130K@50 は未確認（サーバ側は 14 token 生成で完了）
- 128K 超過（prompt > 131,072）: 260K@50 成功、260K@90 成功、480K@50 成功
- **精度差は出ていない**。両側とも、応答本文が取れたセルはすべて正答
- ただし needle の絶対位置で見ると、超過側は 130,086 / 234,155 / 239,999 トークンで、いずれも学習コンテキスト 262,144 の内側にある。YaRN が効くのは 262,144 を超える位置なので、**YaRN の外挿域の recall は未検証**
- 位置 10% が全長で未実施のため、「lost in the middle」の形（中間が苦手）は判定できていない。50% と 90% の比較では差が見えていない

## prefill 速度の深さ依存

| プロンプト長 | engine prefill tok/s | TTFT 秒 | 備考 |
| ---: | ---: | ---: | --- |
| 33K（50%） | 474.2 | 69.99 | prefix 再利用なし |
| 33K（90%） | 422.6 | 39.76 | 16,384 を再利用 |
| 130K（50%） | 393.4 | 163.10 | 65,536 を再利用。読んだ 64,163 トークンだけの速度 |
| 130K（90%） | 418.4 | 271.23 | 16,384 を再利用 |
| 260K（50%） | 365.9 | 667.09 | 16,384 を再利用 |
| 260K（90%） | 368.4 | 662.55 | 16,384 を再利用 |
| 480K（50%） | 302.6 | 1,533.4 | 16,384 を再利用 |

- プロンプトが深いほど prefill が遅い。33K の 474 tok/s から 480K の 303 tok/s まで、長さに対して約 36% 低下する
- KV streaming のヒット率は長いほど上がる（480K で 75.5%、260K で 79.5-82.7%、130K で 83.7-84.9%）。ヒット率が高いのに遅いので、律速は KV 読み出しだけではないと見られる（推定）
- 先行タスクの 300K 実測は 482 tok/s（TTFT 630.7 秒）で、今回の 260K 366 tok/s と整合しない。同じ構成・同じ合成コーパスのはずで、原因は未確認。今回の方が遅い
- 60 分の見積もりはこの低下を織り込んでいなかった。480K の prefill だけで 25.6 分かかり、残り時間を圧迫した

## リスク

- **VRAM 余裕が 289-310 MiB しかない**。expert cache を 10,315 slots まで積んでいるため、長文のたびに #2065 の警告条件（512 MiB 未満）に入る。今回は decode が落ちなかったが、別の負荷や同時リクエストで page out が起きる余地は残る
- 480K の prefill に 25.6 分かかる。実利用で 500K 級を投げると、1 リクエストで 30 分近く待つ。TTFT を短くしたい用途には向かない
- 480K の会話は conversation cache に parking できない（snapshot 7,786 MiB > 枠 7,168 MiB）。同じ長文を続けて投げても prefill は毎回やり直しになる。今回、130K と 33K では 16,384-65,536 トークンの再利用が効いたが、480K では効かない
- 位置 10% と 262K 超の位置が未検証。YaRN の品質は「240K 位置までは正答」までしか言えない
- 合成の一意 ID を探す課題は非常に易しい。意味的な recall、要約、一貫性、推論の品質は測っていない。実利用の品質判断には別の課題が必要
- 1 セル 1 回の試行なので、精度は「失敗しなかった」以上のことは言えない

## 未実施・打ち切り

- 60 分ロックに収まらなかったため、位置グリッドを 7 セルで打ち切った
- 480K の prefill が 25.6 分かかり、その後 260K×2（22 分）、130K×2（7 分）、33K×2（2 分）で枠が埋まった
- 未実施: 33K@10、130K@10、260K@10、480K@10、480K@90。480K@90 は YaRN 外挿域（needle が 262K 超）の確認に必要だったが時間切れ
- 128 トークンの decode 計測も未実施。retrieval 応答が 14 トークンで EOS になるため、別途「長く書かせる」プロンプトが必要だった
- ロックは 63.5 分で 60 分を 3.5 分超過した。最後の 130K@50 と 33K@90/50 の完了を待ったため。開始時点の見積もりが prefill 速度の低下を織り込んでいなかった

## 本番の状態

- 計測は HTTP のみ。コンテナの停止・再起動はしていない。`StartedAt` は 2026-11-07T05:59:40Z のまま
- ロック解放時（07:20:53）の確認: `healthy`、`~/Strata/docker/compose/.env` と `~/opt-work/prod.env` は一致、engine の `STRATA_*` は `STRATA_DENSE_F32=1` と rope 2 項目のみ（起動時と同じ）
- 解放の 3 秒後に別タスクがロックを取得し（owner `fusion-block`、07:20:56）、07:21:03 に本番 `strata` を停止して `strata-fusion` を起動した。**本番の停止はこの別タスクによるもので、こちらは関与していない**。復元はそのタスクの手順に従う
- したがって「本番が 512K のまま healthy」の最終確認はロック解放時点（07:20:53）のもの。現在は別タスクが GPU を占有している

## 再現手順

```sh
# GPU ロック
mkdir /tmp/strata-gpu.lock && echo "quality512k $(date -Is)" > /tmp/strata-gpu.lock/owner

# API キーは表示しない
export STRATA_API_KEY=$(grep ^API_KEY= ~/Strata/docker/compose/.env | cut -d= -f2)

# VRAM/RAM の監視（別シェル）
~/work/results/quality/monitor.sh ~/work/results/quality/monitor-seq.log &

# 1 セル: 目標トークン数、needle 位置 %、ID、max_tokens、出力ファイル
python3 ~/work/results/quality/needle.py 492000 50 NEEDLE-7F3A-1928 64 raw-500k-50.jsonl 3.25 retrieval

# 位置グリッドをまとめて回す場合は run_all.sh（500K -> 256K -> 128K -> 32K の順）
~/work/results/quality/run_all.sh

# engine 側の根拠（コンテナ稼働中）
sudo docker exec strata grep -E "prompt [0-9]+ tokens|KV streaming|WARNING: only|stall" \
  /data/state/strata/launch/docker.engine.log | tail -20

# 終了確認とロック解放
sudo docker inspect -f '{{.State.Health.Status}} {{.State.StartedAt}}' strata
diff ~/Strata/docker/compose/.env ~/opt-work/prod.env && echo identical
rm -rf /tmp/strata-gpu.lock
```

- `needle.py` の第 6 引数は文字/token 比（既定 3.25）。実測は 3.22-3.33 だった
- 第 7 引数に `long` を渡すと「ID を 1 行目に書いたうえで 200 語の説明を書く」質問に変わる。128 トークンの decode 計測用（今回は未使用）

## 成果物

- 生データ: `~/work/results/quality/`
  - `run.out`: 全セルの JSON（ttft、prefill、decode、応答本文、found）
  - `raw-*.jsonl`: セルごとの JSON
  - `seq.log`: 各セルの開始・終了時刻
  - `monitor-seq.log`: 5 秒間隔の VRAM/RAM
  - `logs-*.txt`: セルごとの engine ログ抜粋
  - `baseline-state.txt`、`baseline-env.diff`、`baseline-engine-env.txt`: 計測前の状態
- スクリプト: `needle.py`、`monitor.sh`、`run_all.sh`、`run_phase2.sh`、`run_phase3.sh`、`run_phase4.sh`、`prefix_test.py`
