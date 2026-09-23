"""Citation QA: every \\cite key in paper/main.tex exists in paper/refs.bib, and every bib entry is cited.
  python scripts/cite_check.py
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "paper/main.tex").read_text(); bib = (ROOT / "paper/refs.bib").read_text()
cited = set()
for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}", tex):
    cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
missing = sorted(cited - defined); unused = sorted(defined - cited)
print(f"{len(cited)} keys cited, {len(defined)} defined; missing: {missing or 'none'}; uncited: {unused or 'none'}")
sys.exit(1 if missing else 0)
