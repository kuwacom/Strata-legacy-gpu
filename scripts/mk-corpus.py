"""重複した md を内容ハッシュで除いて 1 本のコーパスにする（長文プロンプト用）。

同じ文書が worktree 間で複製されているため、そのままだと suffix-draft が不自然に当たる。
"""
import glob
import hashlib
import os
import sys

out = sys.argv[1]
patterns = [
    os.path.expanduser("~/Strata/docs/*.md"),
    os.path.expanduser("~/work/wt/*/docs/**/*.md"),
]
seen = set()
parts = []
for pat in patterns:
    for p in sorted(glob.glob(pat, recursive=True)):
        try:
            b = open(p, "rb").read()
        except OSError:
            continue
        h = hashlib.md5(b).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        parts.append(b.decode("utf-8", "ignore"))
text = "\n\n".join(parts)
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write(text)
print(f"corpus: {len(seen)} unique files, {len(text)} chars -> {out}")

