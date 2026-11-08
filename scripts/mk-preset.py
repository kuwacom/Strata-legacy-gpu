"""prod の preset JSON を基に、アーム用の preset を作る。

使い方: mk-preset.py <prod-preset.json> <out.json> <arm>
  base  : prod と同一（MTP, spec 4, spec-min-p 0.5, kv-resident 32768, expert-cache auto）
  df2   : dflash2 に差し替え（MTP を外し、--draft dflash2 / --spec 6 / --draft-conf 0.5）
  df2s4 : dflash2 だが窓は prod と同じ 4（drafter だけの A/B）
  kv64  : kv-resident 65536
  kv4k  : kv-resident 4096
  kv32e : kv-resident 32768 + expert-cache を auto より少し下げる（9500）
すべて STRATA_DENSE_F32=1 と STRATA_DECODE_TIMING=1 を env に入れる。
"""
import json
import sys

src, dst, arm = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(src))
args = list(d["args"])

DF2_MODEL = "/data/Strata-data/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack"
MTP_VOCAB = "/data/Strata-data/mtp/rt/draft_vocab.bin"


def drop(flag):
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]


def setflag(flag, value):
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.append(flag)
        args.append(str(value))


if arm == "base":
    pass
elif arm in ("df2", "df2s4"):
    drop("--mtp")
    drop("--spec-min-p")
    setflag("--spec", "6" if arm == "df2" else "4")
    args += ["--draft", "dflash2", "--draft-model", DF2_MODEL, "--draft-fallback", "none",
             "--mtp-draft-vocab", MTP_VOCAB, "--draft-conf", "0.5"]
elif arm == "kv64":
    setflag("--kv-resident", "65536")
elif arm == "kv4k":
    setflag("--kv-resident", "4096")
elif arm == "kv32e":
    setflag("--expert-cache", "9500")
else:
    raise SystemExit(f"unknown arm {arm}")

d["args"] = args
d["log"] = f"/data/state/strata/launch/{arm}.engine.log"
env = dict(d.get("env") or {})
env["STRATA_DENSE_F32"] = "1"
env["STRATA_DECODE_TIMING"] = "1"
d["env"] = env
d.pop("api_key", None)   # API キーは preset に書かない（起動時に env から渡す）
json.dump(d, open(dst, "w"), indent=1)
print(f"wrote {dst}: {len(args)} args, arm {arm}")

