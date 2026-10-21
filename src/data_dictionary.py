"""Step 2b - Write data/DATA_DICTIONARY.md and an .xlsx copy of the cleaned data with a dictionary sheet."""
import pandas as pd
from common import CLEAN_CSV, PROC, ROOT

DICT = [
    ("tie_id", "string", "Unique tie identifier from the raw file: <competition>-<season>-<round>-<nn> (KPO = knockout play-off)."),
    ("season", "string", "Season, e.g. 2018/19."),
    ("season_start", "integer", "Calendar year in which the season started (2012-2025)."),
    ("competition", "category", "UCL = UEFA Champions League; UEL = UEFA Europa League."),
    ("round_original", "string", "Round name exactly as in the raw file."),
    ("round", "category", "Harmonised round: Round of 32, Knockout play-offs, Round of 16, Quarter-finals, Semi-finals."),
    ("round_order", "integer", "1 = entry knockout round (R32 / play-offs), 2 = R16, 3 = QF, 4 = SF."),
    ("entry_round", "0/1", "1 if the tie is in the entry knockout round (UEL R32 pre-2021; knockout play-offs post-2021)."),
    ("rule_era", "category", "pre = away-goals rule active (2012/13-2019/20); post = abolished (2021/22-2025/26)."),
    ("post", "0/1", "1 if rule_era == post."),
    ("league_phase_format", "0/1", "1 for 2024/25 and 2025/26 (36-team league-phase format)."),
    ("leg1_home_team", "string", "Club hosting the first leg (harmonised name)."),
    ("leg2_home_team", "string", "Club hosting the second (deciding) leg."),
    ("leg1_home_goals", "integer", "Goals by the first-leg host in leg 1."),
    ("leg1_away_goals", "integer", "Goals by the first-leg visitor (= second-leg host) in leg 1."),
    ("leg2_home_goals_final", "integer", "Goals by the second-leg host in leg 2, including extra time."),
    ("leg2_away_goals_final", "integer", "Goals by the second-leg visitor in leg 2, including extra time."),
    ("extra_time", "0/1", "1 if extra time was played."),
    ("penalties", "0/1", "1 if the tie was decided by a penalty shoot-out."),
    ("winner", "string", "Club that advanced."),
    ("leg2_home_goals_90", "integer", "Second-leg host goals after 90 minutes (see leg2_90min_source)."),
    ("leg2_away_goals_90", "integer", "Second-leg visitor goals after 90 minutes."),
    ("leg2_90min_source", "string", "How the 90-minute score was obtained: no extra time / derived (pre-abolition ET condition) / openfootball 90-min score / sourced file."),
    ("leg1_total_goals", "integer", "Total goals in leg 1."),
    ("leg2_total_goals_90", "integer", "Total goals in leg 2 over 90 minutes (H1 outcome)."),
    ("leg2_total_goals_final", "integer", "Total goals in leg 2 including extra time."),
    ("extra_time_goals", "integer", "Goals scored in extra time (0 if no extra time)."),
    ("tie_total_goals_180", "integer", "Goals over both legs, 180 minutes."),
    ("away_side_goals_180", "integer", "Goals scored by the visiting team in each leg, summed over 180 minutes."),
    ("agg_level_180", "0/1", "1 if the aggregate was level after 180 minutes (H2 outcome; independent of the tie-breaker)."),
    ("away_goals_unequal_180", "0/1", "1 if the two clubs' away-goal counts differed after 180 minutes."),
    ("away_goals_rule_would_decide", "0/1", "1 if level after 180 min with unequal away goals, i.e. the away-goals rule settles the tie (actual pre, counterfactual post)."),
    ("leg2_host_first_leg_margin", "integer", "First-leg goal difference from the second-leg host's perspective (>0 leading, <0 trailing)."),
    ("first_leg_abs_margin", "integer", "Absolute first-leg goal difference."),
    ("deficit_category", "category", "level / 1 goal / 2+ goals."),
    ("leg2_host_position", "category", "leading / level / trailing (second-leg host after leg 1)."),
    ("leg2_host_advanced", "0/1", "1 if the second-leg host advanced (H6 outcome)."),
    ("first_leg_trailer", "string", "Club trailing after leg 1 (empty if leg 1 was drawn)."),
    ("comeback_eligible", "0/1", "1 if leg 1 had a winner."),
    ("comeback", "0/1/blank", "1 if the first-leg trailer advanced by any route (H5 outcome); blank if not eligible."),
    ("trailer_hosts_leg2", "0/1/blank", "1 if the first-leg trailer hosted the second leg; blank if not eligible."),
    ("resolution", "category", "aggregate (180 min) / away goals (180 min) / extra time / penalty shoot-out."),
    ("covid_disrupted_season", "0/1", "1 for 2019/20 (some legs behind closed doors or delayed to August 2020)."),
    ("sample_primary", "0/1", "1 for Round of 16, quarter-finals and semi-finals (primary like-for-like sample; 349 ties)."),
    ("sample_extended", "0/1", "1 for all ties (533)."),
    ("sample_draft_477", "0/1", "1 for the asymmetric sample used in the 2026 draft (all pre ties + primary post ties); sensitivity only."),
    ("notes_original", "string", "Free-text note from the raw file (unchanged)."),
]


def main():
    d = pd.read_csv(CLEAN_CSV)
    assert [c for c, _, _ in DICT] == list(d.columns), "dictionary out of sync with clean data"
    L = ["# Data dictionary: data/processed/uefa_two_legged_ties_clean.csv", "",
         f"{len(d)} rows (one per two-legged tie), {len(d.columns)} columns. Generated by `src/data_dictionary.py`.", "",
         "| Column | Type | Description |", "|---|---|---|"]
    L += [f"| `{c}` | {t} | {desc} |" for c, t, desc in DICT]
    (ROOT / "data" / "DATA_DICTIONARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with pd.ExcelWriter(PROC / "uefa_two_legged_ties_clean.xlsx") as xw:
        d.to_excel(xw, sheet_name="ties", index=False)
        pd.DataFrame(DICT, columns=["column", "type", "description"]).to_excel(xw, sheet_name="data_dictionary", index=False)
    print("data dictionary written")


if __name__ == "__main__":
    main()
