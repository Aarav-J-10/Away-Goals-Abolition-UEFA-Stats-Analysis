# Cleaning log

Input: `data/raw/UEFA_Tie_Data_RAW.xlsx` (533 ties, 20 columns; never modified).
Output: `data/processed/uefa_two_legged_ties_clean.csv` (533 ties, 47 columns), produced by `src/clean.py`.

## 0. Verification before cleaning (`src/verify_raw.py`)

| Check | Result |
|---|---|
| Leg scores matched to an independent source (openfootball for all UCL and UEL 2021/22-2024/25; Wikipedia for UEL 2012/13-2019/20 and 2025/26) | **1,066 / 1,066** legs identical |
| Extra-time flag agreement | 531 / 533; both differences are gaps in openfootball (PSV-Juventus and Inter-Barcelona 2024/25 omit the "a.e.t." marker); the dataset is correct |
| Penalty flag agreement | 533 / 533 |
| Source ties missing from the dataset | 0 (RB Leipzig v Spartak Moscow 2021/22 was never played; correctly absent) |
| Duplicate ties / source legs used twice | 0 / 0 |
| Winner implied by each era's rules = recorded winner | 533 / 533 |
| Notes consistent with scores, extra time and penalties | 533 / 533 |

No score, flag or winner in the raw file needed correcting.

## 1. Structural cleaning

1. Stripped whitespace from column names (raw column `"Leg 2 Home Goals "` had a trailing space) and from all text cells.
2. Renamed columns to snake_case (see `data/DATA_DICTIONARY.md`).
3. Competition names shortened to `UCL` / `UEL`; season written as `2018/19`.
4. Rule era recoded to `pre` / `post` with a 0/1 indicator `post`.
5. Round names harmonised: "Knockout round play-offs" (UEL 2021/22-2023/24) and "Knockout phase play-offs" (UCL/UEL 2024/25-2025/26) both become `Knockout play-offs`; the original label is kept in `round_original`.
6. Club-name variants unified: `Internazionale` -> `Inter Milan`; `Steaua București` -> `FCSB`; `Malaga` -> `Málaga`.
7. Redundant columns `Team A`, `Team B`, `Leg 2 Away` and `Rule Era Label` dropped (they duplicate other fields; the verifier confirms leg 2 is leg 1 reversed for every tie).

## 2. Ninety-minute second-leg scores

The raw file stores second-leg scores **after extra time**. Because extra time is more frequent after abolition by construction,
counting extra-time goals would bias a comparison of second-leg scoring. 90-minute scores were therefore added:

| Source | Ties | Method |
|---|---|---|
| No extra time | 471 | 90-minute score = final score |
| Pre-abolition, extra time | 28 | **Derived exactly**: extra time required level aggregate *and* level away goals, which forces the leg-2 90-minute score to mirror leg 1 |
| Post-abolition, extra time | 27 | openfootball full-time annotation, e.g. `3-1 a.e.t. (2-1, 0-0)` |
| Post-abolition, extra time | 7 | `data/external/leg2_90min_scores_post_abolition_et.csv`, each with its documented source |

`src/clean.py` asserts that, with these scores, extra time occurs in exactly the ties the rules of each era require.

## 3. Derived variables

Goal totals (leg 1, leg 2 at 90 minutes and after extra time, extra-time goals, visiting-team goals), aggregate-level and
away-goal indicators after 180 minutes, the counterfactual "away-goals rule would decide" flag, first-leg margin from the
second-leg host's perspective, deficit category, comeback eligibility and success, second-leg host advancement, resolution
pathway, and sample flags. Definitions are in `data/DATA_DICTIONARY.md`.

## 4. Sample flags (nothing is deleted)

| Flag | Pre | Post | Definition |
|---|---|---|---|
| `sample_primary` | 210 | 139 | Round of 16, quarter-finals, semi-finals (same rounds in both eras) |
| `sample_extended` | 338 | 195 | All ties |
| `sample_draft_477` | 338 | 139 | Sample used in the 2026 draft; asymmetric (keeps pre-abolition R32, drops post-abolition play-offs); sensitivity only |
| `covid_disrupted_season` | 30 | 0 | Season 2019/20 |

## 5. Issues found in the earlier (draft) analysis files

These did not affect the raw data but are corrected in this release:

* The comeback test in the draft used all 533 ties (post-abolition n = 145 eligible ties), while every other test used the
  477-tie sample (post-abolition n = 101 eligible), so the reported comeback figures mixed samples.
* The draft robustness report listed the extra-time rate as 2.66% -> 7.19%; those are the penalty shoot-out rates. The extra-time rates were 8.28% -> 15.11% in that sample.
* The draft's Fisher odds ratio for shoot-outs among extra-time ties (0.52) was oriented pre-vs-post; post-vs-pre it is 1.92.
* Second-leg goals in the draft included extra-time goals.
* The draft's home-advancement model used an unsigned first-leg deficit, which cannot distinguish a second-leg host that leads from one that trails.
* The draft's main sample compared a pre-abolition sample that included the UEL Round of 32 with a post-abolition sample that excluded the equivalent play-off round.
