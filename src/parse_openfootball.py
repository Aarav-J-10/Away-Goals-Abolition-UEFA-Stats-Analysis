"""Parse an openfootball champions-league clone into a flat table of knockout matches.

Usage: python src/parse_openfootball.py <path-to-champions-league-clone> <out.csv>
Only needed to refresh data/external/openfootball_knockout_matches.csv."""
import re, sys, glob, os, unicodedata
import pandas as pd

ROOT = sys.argv[1]
OUT = sys.argv[2]

line_re = re.compile(r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(?P<h>.+?\(\w{2,4}\))\s+v\s+(?P<a>.+?\(\w{2,4}\))\s+(?P<s>\S.*)$")
pen_re = re.compile(r"(\d+)-(\d+)\s+pen\.\s+(\d+)-(\d+)\s+a\.e\.t\.\s*\((\d+)-(\d+)")
aet_re = re.compile(r"(\d+)-(\d+)\s+a\.e\.t\.\s*\((\d+)-(\d+)")
plain_re = re.compile(r"^(\d+)-(\d+)")

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.txt"))):
    season = os.path.basename(os.path.dirname(f))
    comp = os.path.basename(f).replace(".txt", "")
    if comp not in ("cl", "el"):
        continue
    section = None
    for ln in open(f, encoding="utf-8"):
        if ln.startswith("▪"):
            section = ln.strip("▪ \n")
            continue
        if section is None or re.search(r"Group|Gruppe|League", section):
            continue
        m = line_re.match(ln.rstrip())
        if not m:
            continue
        h, a, s = m.group("h").strip(), m.group("a").strip(), m.group("s")
        rec = dict(season=season, comp=comp, section=section, home=h, away=a, raw=s.strip(),
                   hg_final=None, ag_final=None, hg_90=None, ag_90=None, aet=False, pens=False,
                   pen_h=None, pen_a=None)
        p = pen_re.search(s); e = aet_re.search(s); q = plain_re.match(s)
        if p:
            rec.update(pen_h=int(p[1]), pen_a=int(p[2]), hg_final=int(p[3]), ag_final=int(p[4]),
                       hg_90=int(p[5]), ag_90=int(p[6]), aet=True, pens=True)
        elif e:
            rec.update(hg_final=int(e[1]), ag_final=int(e[2]), hg_90=int(e[3]), ag_90=int(e[4]), aet=True)
        elif q:
            rec.update(hg_final=int(q[1]), ag_final=int(q[2]), hg_90=int(q[1]), ag_90=int(q[2]))
        else:
            rec["unparsed"] = True
        rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.groupby(["season", "comp"]).size().to_string())
print("unparsed:", df.get("unparsed", pd.Series(dtype=bool)).fillna(False).sum())
