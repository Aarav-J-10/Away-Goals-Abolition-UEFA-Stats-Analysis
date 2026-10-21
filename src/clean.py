"""Step 2 - Build the cleaned, analysis-ready tie table from the verified raw file.

Every transformation is listed in docs/CLEANING_LOG.md. Nothing in the raw file is overwritten.
Output: data/processed/uefa_two_legged_ties_clean.csv (+ .xlsx with a data-dictionary sheet)
"""
import pandas as pd
import numpy as np
from common import RAW, EXT, PROC, CLEAN_CSV

TEAM_CANON = {"Internazionale": "Inter Milan", "Steaua București": "FCSB", "Malaga": "Málaga"}
ROUND_CANON = {"Round of 32": "Round of 32", "Knockout round play-offs": "Knockout play-offs",
               "Knockout phase play-offs": "Knockout play-offs", "Round of 16": "Round of 16",
               "Quarter-finals": "Quarter-finals", "Semi-finals": "Semi-finals"}
ROUND_ORDER = {"Round of 32": 1, "Knockout play-offs": 1, "Round of 16": 2, "Quarter-finals": 3, "Semi-finals": 4}


def main():
    raw = pd.read_excel(RAW)
    raw.columns = [c.strip() for c in raw.columns]
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].astype(str).str.strip()
    d = pd.DataFrame()
    d["tie_id"] = raw["Tie ID"]
    d["season"] = raw["Season"].str.replace("-", "/", regex=False)
    d["season_start"] = raw["Season"].str[:4].astype(int)
    d["competition"] = raw["Competition"].map({"UEFA Champions League": "UCL", "UEFA Europa League": "UEL"})
    d["round_original"] = raw["Round"]
    d["round"] = raw["Round"].map(ROUND_CANON)
    d["round_order"] = d["round"].map(ROUND_ORDER)
    d["entry_round"] = (d.round_order == 1).astype(int)
    d["rule_era"] = raw["Rule Era"].map({"Pre-abolition": "pre", "Post-abolition": "post"})
    d["post"] = (d.rule_era == "post").astype(int)
    d["league_phase_format"] = (d.season_start >= 2024).astype(int)  # 36-team league phase from 2024/25

    canon = lambda s: s.map(lambda x: TEAM_CANON.get(x, x))
    d["leg1_home_team"] = canon(raw["Leg 1 Home"])   # hosts leg 1
    d["leg2_home_team"] = canon(raw["Leg 2 Home"])   # hosts the deciding second leg
    d["leg1_home_goals"] = raw["Leg 1 Home Goals"].astype(int)
    d["leg1_away_goals"] = raw["Leg 1 Away Goals"].astype(int)
    d["leg2_home_goals_final"] = raw["Leg 2 Home Goals"].astype(int)
    d["leg2_away_goals_final"] = raw["Leg 2 Away Goals"].astype(int)
    d["extra_time"] = (raw["Extra Time?"] == "Yes").astype(int)
    d["penalties"] = (raw["Penalties?"] == "Yes").astype(int)
    d["winner"] = canon(raw["Winner"])

    # ---- 90-minute second-leg score
    # Pre-abolition: extra time was played only if aggregate AND away goals were level after 180 min,
    # which forces the leg-2 90-min score to mirror leg 1 (host scores leg1_home_goals, visitor scores leg1_away_goals).
    # Post-abolition: taken from openfootball's half/full-time annotation or from the sourced file.
    of = pd.read_csv(EXT / "openfootball_knockout_matches.csv")
    xc = pd.read_csv(EXT / "verification_leg_crosscheck.csv").set_index("tie_id")
    manual = pd.read_csv(EXT / "leg2_90min_scores_post_abolition_et.csv").set_index("tie_id")
    h90, a90, src90 = [], [], []
    for _, r in d.iterrows():
        if not r.extra_time:
            h90.append(r.leg2_home_goals_final); a90.append(r.leg2_away_goals_final); src90.append("no extra time")
        elif r.rule_era == "pre":
            h90.append(r.leg1_home_goals); a90.append(r.leg1_away_goals); src90.append("derived (pre-abolition ET condition)")
        elif r.tie_id in manual.index:
            h90.append(int(manual.loc[r.tie_id, "leg2_home_goals_90"])); a90.append(int(manual.loc[r.tie_id, "leg2_away_goals_90"]))
            src90.append("sourced file")
        else:
            x = xc.loc[r.tie_id]
            m = of[(of.home == x.leg2_src_home) & (of.away == x.leg2_src_away) & (of.season == r.season.replace("/", "-"))].iloc[0]
            assert m.aet, r.tie_id
            h90.append(int(m.hg_90)); a90.append(int(m.ag_90)); src90.append("openfootball 90-min score")
    d["leg2_home_goals_90"], d["leg2_away_goals_90"], d["leg2_90min_source"] = h90, a90, src90

    # ---- derived goal variables
    d["leg1_total_goals"] = d.leg1_home_goals + d.leg1_away_goals
    d["leg2_total_goals_90"] = d.leg2_home_goals_90 + d.leg2_away_goals_90
    d["leg2_total_goals_final"] = d.leg2_home_goals_final + d.leg2_away_goals_final
    d["extra_time_goals"] = d.leg2_total_goals_final - d.leg2_total_goals_90
    d["tie_total_goals_180"] = d.leg1_total_goals + d.leg2_total_goals_90
    d["away_side_goals_180"] = d.leg1_away_goals + d.leg2_away_goals_90   # goals scored by visiting teams over both legs

    # perspective of the second-leg host (H2) and the first-leg host (H1)
    agg_h2 = d.leg1_away_goals + d.leg2_home_goals_90
    agg_h1 = d.leg1_home_goals + d.leg2_away_goals_90
    d["agg_level_180"] = (agg_h2 == agg_h1).astype(int)
    d["away_goals_unequal_180"] = (d.leg2_away_goals_90 != d.leg1_away_goals).astype(int)
    # would the away-goals rule have settled this tie after 180 min? (actual for pre, counterfactual for post)
    d["away_goals_rule_would_decide"] = (d.agg_level_180 & d.away_goals_unequal_180).astype(int)
    d["leg2_host_first_leg_margin"] = d.leg1_away_goals - d.leg1_home_goals  # >0: second-leg host leads after leg 1
    d["first_leg_abs_margin"] = d.leg2_host_first_leg_margin.abs()
    d["deficit_category"] = pd.cut(d.first_leg_abs_margin, [-1, 0, 1, 99], labels=["level", "1 goal", "2+ goals"]).astype(str)
    d["leg2_host_position"] = np.select([d.leg2_host_first_leg_margin > 0, d.leg2_host_first_leg_margin < 0],
                                        ["leading", "trailing"], "level")
    d["leg2_host_advanced"] = (d.winner == d.leg2_home_team).astype(int)
    trailer = np.where(d.leg2_host_first_leg_margin < 0, d.leg2_home_team,
                       np.where(d.leg2_host_first_leg_margin > 0, d.leg1_home_team, ""))
    d["first_leg_trailer"] = trailer
    d["comeback_eligible"] = (d.first_leg_abs_margin > 0).astype(int)
    d["comeback"] = np.where(d.comeback_eligible == 1, (d.winner == d.first_leg_trailer).astype(int), np.nan)
    d["trailer_hosts_leg2"] = np.where(d.comeback_eligible == 1, (d.leg2_host_first_leg_margin < 0).astype(int), np.nan)

    # how the tie was resolved
    def resolution(r):
        if r.penalties:
            return "penalty shoot-out"
        if r.extra_time:
            return "extra time"
        if r.agg_level_180:
            return "away goals (180 min)"
        return "aggregate (180 min)"
    d["resolution"] = d.apply(resolution, axis=1)

    # ---- samples & flags
    d["covid_disrupted_season"] = (d.season == "2019/20").astype(int)
    d["sample_primary"] = d["round"].isin(["Round of 16", "Quarter-finals", "Semi-finals"]).astype(int)
    d["sample_extended"] = 1
    d["sample_draft_477"] = ((d.rule_era == "pre") | (d.sample_primary == 1)).astype(int)  # sample used in the 2026 draft
    d["notes_original"] = raw["Notes"]

    # ---- sanity assertions
    assert d.tie_id.is_unique and len(d) == 533
    assert ((d.extra_time == 1) == ((d.agg_level_180 == 1) & ((d.rule_era == "post") | (d.away_goals_unequal_180 == 0)))).all()
    assert (d.extra_time_goals >= 0).all()
    assert ((d.penalties == 0) | (d.extra_time == 1)).all()
    assert d.loc[d.comeback_eligible == 1, "first_leg_trailer"].ne("").all()

    d.to_csv(CLEAN_CSV, index=False)
    print(d.groupby(["rule_era", "round"]).size().unstack(0))
    print("primary:", d.groupby("rule_era").sample_primary.sum().to_dict(), " extended:", d.rule_era.value_counts().to_dict())
    print(d.leg2_90min_source.value_counts())
    print(d.resolution.groupby(d.rule_era).value_counts().unstack(0))


if __name__ == "__main__":
    main()
