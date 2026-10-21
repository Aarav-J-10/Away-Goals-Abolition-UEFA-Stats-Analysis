"""Step 1 - Verify the raw tie file before anything is cleaned or analysed.

Checks
  A. Every leg score, extra-time flag and penalty flag against an independent source:
       * openfootball/champions-league (CC0): all UCL seasons, UEL 2021/22-2024/25
       * Wikipedia knockout-phase pages (transcribed in data/external/wikipedia_uel): UEL 2012/13-2019/20, 2025/26
  B. Completeness: every two-legged knockout tie in the sources appears exactly once.
  C. Internal logic: leg/team consistency, winner implied by the rules of each era,
     extra-time/penalty logic, and agreement of the free-text Notes with the scores.

Outputs: data/external/verification_leg_crosscheck.csv, docs/VERIFICATION_REPORT.md
Exit code 1 if any unexplained discrepancy remains.
"""
import csv, glob, os, re, sys, unicodedata
from difflib import SequenceMatcher
import pandas as pd
from common import RAW, EXT, DOCS

# Ties where the external source itself is incomplete (documented, dataset value confirmed elsewhere)
KNOWN_SOURCE_GAPS = {
    "UCL-2024-25-KPO-04": "openfootball omits the a.e.t. marker for PSV 3-1 Juventus (extra time confirmed; see leg2_90min_scores_post_abolition_et.csv)",
    "UCL-2024-25-SF-02": "openfootball omits the a.e.t. marker for Inter 4-3 Barcelona (extra time confirmed; see leg2_90min_scores_post_abolition_et.csv)",
}

STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "rsc", "sv", "vfl", "vfb", "sl", "sk", "fk", "nk", "gnk", "cd",
        "rc", "club", "de", "real", "1", "04", "05", "1899", "1909", "1900", "1913", "tsg", "bv", "bsc", "pfc", "hnk",
        "ogc", "us", "ss", "aca", "spor", "kulubu", "calcio", "olympique"}
ALIAS = {"internazionale": "inter", "inter milan": "inter", "fc internazionale milano": "inter",
         "bayern munich": "bayern munchen", "psv eindhoven": "psv", "club atletico de madrid": "atletico madrid",
         "spartak moscow": "spartak moskva", "zenit saint petersburg": "zenit", "red star belgrade": "crvena zvezda",
         "dynamo kyiv": "dynamo kiev", "copenhagen": "kobenhavn", "losc lille": "lille",
         "sporting clube de portugal": "sporting cp", "olympiakos piraeus": "olympiacos",
         "pae olympiakos sfp": "olympiacos", "stade rennais": "rennes", "red bull salzburg": "salzburg",
         "rb salzburg": "salzburg", "fc red bull salzburg": "salzburg"}


def norm(s):
    s = re.sub(r"\(\w{2,4}\)", "", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())
    s = ALIAS.get(s, s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks) if toks else s


def sim(a, b):
    a, b = norm(a), norm(b)
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


def load_sources():
    of = pd.read_csv(EXT / "openfootball_knockout_matches.csv")
    of["source"] = "openfootball"
    of["cancelled"] = of.raw.astype(str).str.contains("cancelled")
    rows = []
    for f in sorted(glob.glob(str(EXT / "wikipedia_uel" / "uel_*.csv"))):
        season = os.path.basename(f)[4:11]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            rows.append(dict(season=season, comp="el", section=r["round"], home=r["home_team"], away=r["away_team"],
                             raw=f'{r["home_goals"]}-{r["away_goals"]} {r["extra_time"]} {r["shootout"]}'.strip(),
                             hg_final=int(r["home_goals"]), ag_final=int(r["away_goals"]),
                             aet=r["extra_time"].strip() == "yes", pens=bool(r["shootout"].strip()),
                             source="wikipedia"))
    w = pd.DataFrame(rows); w["cancelled"] = False
    return pd.concat([of, w], ignore_index=True)


def main():
    raw = pd.read_excel(RAW)
    raw.columns = [c.strip() for c in raw.columns]
    src = load_sources()
    COMP = {"UEFA Champions League": "cl", "UEFA Europa League": "el"}

    # ---- A. leg-by-leg cross-check
    recs = []
    for _, r in raw.iterrows():
        pool = src[(src.season == r.Season) & (src.comp == COMP[r.Competition]) & ~src.cancelled]
        out = {"tie_id": r["Tie ID"], "season": r.Season}
        for leg in (1, 2):
            h, a = r[f"Leg {leg} Home"], r[f"Leg {leg} Away"]
            sc = pool.apply(lambda m: min(sim(h, m.home), sim(a, m.away)), axis=1)
            m = pool.loc[sc.idxmax()]
            out.update({f"leg{leg}_source": m.source, f"leg{leg}_name_match": round(sc.max(), 2),
                        f"leg{leg}_src_home": m.home, f"leg{leg}_src_away": m.away, f"leg{leg}_src_raw": m.raw,
                        f"leg{leg}_dataset_score": f'{r[f"Leg {leg} Home Goals"]}-{r[f"Leg {leg} Away Goals"]}',
                        f"leg{leg}_score_ok": int(m.hg_final) == r[f"Leg {leg} Home Goals"] and int(m.ag_final) == r[f"Leg {leg} Away Goals"]})
            if leg == 2:
                out["et_ok"] = bool(m.aet) == (r["Extra Time?"] == "Yes")
                out["pen_ok"] = bool(m.pens) == (r["Penalties?"] == "Yes")
        out["all_ok"] = out["leg1_score_ok"] and out["leg2_score_ok"] and out["et_ok"] and out["pen_ok"] \
            and out["leg1_name_match"] >= 0.8 and out["leg2_name_match"] >= 0.8
        out["note"] = KNOWN_SOURCE_GAPS.get(r["Tie ID"], "")
        recs.append(out)
    xc = pd.DataFrame(recs)
    xc.to_csv(EXT / "verification_leg_crosscheck.csv", index=False)
    unexplained = xc[~xc.all_ok & (xc.note == "")]

    # ---- B. completeness
    used = set(zip(xc.season, xc.leg1_src_home, xc.leg1_src_away)) | set(zip(xc.season, xc.leg2_src_home, xc.leg2_src_away))
    dup_use = pd.concat([xc[["season", "leg1_src_home", "leg1_src_away"]].set_axis(["s", "h", "a"], axis=1),
                         xc[["season", "leg2_src_home", "leg2_src_away"]].set_axis(["s", "h", "a"], axis=1)]).duplicated().sum()
    sk = src[src.season.isin(raw.Season.unique()) & ~src.section.isin(["Final", "Finals, Final"])]
    sk = sk[~((sk.season == "2019-20") & (sk.comp == "cl") & sk.section.isin(["Quarterfinals", "Semifinals"]))]
    missing = sk[~sk.apply(lambda m: (m.season, m.home, m.away) in used, axis=1)]

    # ---- C. internal logic
    issues = []
    if raw["Tie ID"].duplicated().any():
        issues.append("duplicate Tie IDs")
    for _, r in raw.iterrows():
        tid = r["Tie ID"]
        if not (r["Leg 1 Home"] == r["Leg 2 Away"] and r["Leg 1 Away"] == r["Leg 2 Home"]):
            issues.append(f"{tid}: leg 2 teams are not leg 1 reversed")
        A, B = r["Leg 1 Home"], r["Leg 1 Away"]
        a1, b1, b2, a2 = r["Leg 1 Home Goals"], r["Leg 1 Away Goals"], r["Leg 2 Home Goals"], r["Leg 2 Away Goals"]
        aggA, aggB = a1 + a2, b1 + b2
        et, pen = r["Extra Time?"] == "Yes", r["Penalties?"] == "Yes"
        pre = r["Rule Era"] == "Pre-abolition"
        if pen and not et:
            issues.append(f"{tid}: penalties without extra time")
        if aggA != aggB:
            w = A if aggA > aggB else B
        elif pre and a2 != b1:
            w = A if a2 > b1 else B
        else:
            w = None
        if w is None and not pen:
            issues.append(f"{tid}: tie undecided but no shoot-out")
        if w is not None and pen:
            issues.append(f"{tid}: shoot-out recorded but tie decided")
        if w is not None and w != r.Winner:
            issues.append(f"{tid}: winner should be {w}")
        note = str(r.Notes)
        m = re.search(r"(\d+)-(\d+) aggregate", note) or re.search(r"won (\d+)-(\d+) on aggregate", note)
        if m and sorted([int(m[1]), int(m[2])]) != sorted([aggA, aggB]):
            issues.append(f"{tid}: note aggregate disagrees with scores")
        if ("penalties" in note) != pen or (("extra time" in note) or ("penalties" in note)) != et:
            issues.append(f"{tid}: note disagrees with ET/penalty flags")
        if "away goals" in note and not pre:
            issues.append(f"{tid}: away goals cited in post-abolition tie")

    # ---- report
    n = len(raw)
    by_src = xc.leg1_source.value_counts().to_dict()
    lines = ["# Verification report for data/raw/UEFA_Tie_Data_RAW.xlsx", "",
             "Generated by `src/verify_raw.py`.", "",
             "## A. Leg-by-leg cross-check against independent sources", "",
             f"* Ties checked: **{n}** ({2*n} legs). Source of first-leg check: {by_src}.",
             f"* Legs whose score matched the source exactly: **{int(xc.leg1_score_ok.sum() + xc.leg2_score_ok.sum())} / {2*n}**.",
             f"* Extra-time flag agreement: **{int(xc.et_ok.sum())} / {n}**; penalty flag agreement: **{int(xc.pen_ok.sum())} / {n}**.",
             f"* Discrepancies explained by gaps in the source (dataset correct): {len(KNOWN_SOURCE_GAPS)}"]
    for k, v in KNOWN_SOURCE_GAPS.items():
        lines.append(f"    * `{k}`: {v}")
    lines += [f"* **Unexplained discrepancies: {len(unexplained)}**", "",
              "## B. Completeness", "",
              f"* Two-legged knockout legs in the sources for the study seasons: {len(sk)}.",
              f"* Source legs not present in the dataset: {len(missing)}"]
    for _, m in missing.iterrows():
        lines.append(f"    * {m.season} {m.comp} {m.section}: {m.home} v {m.away} [{m.raw}]")
    lines += [f"* Source legs matched to more than one dataset leg: {dup_use}", "",
              "The only source tie absent from the dataset is RB Leipzig v Spartak Moscow (UEL 2021/22 Round of 16),",
              "which was never played (Spartak were suspended; Leipzig advanced by walkover). It is correctly excluded.", "",
              "## C. Internal logic", "",
              f"* Logic violations: **{len(issues)}**"] + [f"    * {i}" for i in issues]
    (DOCS / "VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    ok = len(unexplained) == 0 and len(issues) == 0 and dup_use == 0 and \
        all("cancelled" in str(x) for x in missing.raw)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
