"""Step 3 - Pre-specified statistical analysis.

Primary sample  : Round of 16, quarter-finals and semi-finals of the UCL and UEL (rounds that exist, in the
                  same form, in both eras); 210 pre-abolition and 139 post-abolition ties.
Extended sample : all 533 two-legged knockout ties, including entry rounds (UEL Round of 32 before 2021;
                  knockout play-offs after), with round controls.

Primary hypothesis family (Holm-adjusted, alpha = 0.05):
  H1 second-leg goals in 90 minutes      H4 penalty shoot-out incidence
  H2 aggregate level after 180 minutes   H5 comeback by the first-leg trailer
  H3 extra-time incidence                H6 advancement of the second-leg host

Outputs: results/results.json, results/tables/*.csv, results/analysis_output.md
"""
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
from common import CLEAN_CSV, TAB, RES, SEED
from firth import firth_logit

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
B_BOOT = 10000
B_PERM = 10000
R = {}  # everything that is reported goes in here


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(round(x, 4))


# --------------------------------------------------------------------------------------------- tests
def two_prop(s, col):
    pre, post = s[s.post == 0][col].dropna(), s[s.post == 1][col].dropna()
    x1, n1, x2, n2 = int(post.sum()), len(post), int(pre.sum()), len(pre)
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    pz = 2 * stats.norm.sf(abs(z))
    lo, hi = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb", compare="diff")
    orr, pf = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
    ci_or = stats.contingency.odds_ratio([[x1, n1 - x1], [x2, n2 - x2]]).confidence_interval(0.95)
    w1 = proportion_confint(x1, n1, method="wilson"); w2 = proportion_confint(x2, n2, method="wilson")
    return dict(pre_x=x2, pre_n=n2, pre_p=r4(p2), pre_ci=[r4(w2[0]), r4(w2[1])],
                post_x=x1, post_n=n1, post_p=r4(p1), post_ci=[r4(w1[0]), r4(w1[1])],
                diff_pp=r4(100 * (p1 - p2)), diff_ci_pp=[r4(100 * lo), r4(100 * hi)], z=r4(z), p_z=r4(pz),
                fisher_or=r4(orr), fisher_or_ci=[r4(ci_or.low), r4(ci_or.high)], p_fisher=r4(pf),
                cohens_h=r4(proportion_effectsize(p1, p2)))


def mean_diff(s, col, sesoi=0.5):
    pre, post = s[s.post == 0][col].values.astype(float), s[s.post == 1][col].values.astype(float)
    t = stats.ttest_ind(post, pre, equal_var=False)
    ci = t.confidence_interval(0.95)
    v1, v2 = post.var(ddof=1) / len(post), pre.var(ddof=1) / len(pre)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (len(post) - 1) + v2 ** 2 / (len(pre) - 1))
    se = np.sqrt(v1 + v2)
    diff = post.mean() - pre.mean()
    # TOST equivalence test with bounds +/- sesoi goals
    p_lower = stats.t.sf((diff + sesoi) / se, df)
    p_upper = stats.t.cdf((diff - sesoi) / se, df)
    mw = stats.mannwhitneyu(post, pre, alternative="two-sided")
    rb = 2 * mw.statistic / (len(post) * len(pre)) - 1
    pooled = np.sqrt(((len(post) - 1) * post.var(ddof=1) + (len(pre) - 1) * pre.var(ddof=1)) / (len(post) + len(pre) - 2))
    allv = np.concatenate([post, pre]); n1 = len(post)
    perm = np.array([allv[idx[:n1]].mean() - allv[idx[n1:]].mean()
                     for idx in (rng.permutation(len(allv)) for _ in range(B_PERM))])
    p_perm = (np.sum(np.abs(perm) >= abs(diff) - 1e-12) + 1) / (B_PERM + 1)
    return dict(pre_mean=r4(pre.mean()), pre_sd=r4(pre.std(ddof=1)), pre_n=len(pre),
                post_mean=r4(post.mean()), post_sd=r4(post.std(ddof=1)), post_n=len(post),
                diff=r4(diff), diff_ci=[r4(ci.low), r4(ci.high)], welch_t=r4(t.statistic), welch_df=r4(df),
                p_welch=r4(t.pvalue), cohens_d=r4(diff / pooled), mw_U=r4(mw.statistic), p_mw=r4(mw.pvalue),
                rank_biserial=r4(rb), p_perm=r4(p_perm), tost_bound=sesoi, p_tost=r4(max(p_lower, p_upper)))


# --------------------------------------------------------------------------------------------- models
def tidy(res, exp=True, label=""):
    ci = res.conf_int()
    rows = []
    for t in res.params.index:
        rows.append(dict(model=label, term=t, coef=r4(res.params[t]), se=r4(res.bse[t]),
                         ratio=r4(np.exp(res.params[t])) if exp else None,
                         ci_low=r4(np.exp(ci.loc[t, 0]) if exp else ci.loc[t, 0]),
                         ci_high=r4(np.exp(ci.loc[t, 1]) if exp else ci.loc[t, 1]), p=r4(res.pvalues[t])))
    return rows


def logit(formula, data, label, cluster=None):
    m = smf.glm(formula, data, family=sm.families.Binomial())
    res = m.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]}) if cluster else m.fit(cov_type="HC1")
    return res, tidy(res, label=label)


def era_row(rows, term="post"):
    r = [x for x in rows if x["term"] == term][0]
    return dict(ratio=r["ratio"], ci=[r["ci_low"], r["ci_high"]], p=r["p"], coef=r["coef"])


def firth_design(formula, data):
    y, X = __import__("patsy").dmatrices(formula, data, return_type="dataframe")
    return X, y.values.ravel()


# --------------------------------------------------------------------------------------------- counterfactual
def counterfactual(s):
    """Apply the opposite tie-break rule to each era's observed scorelines.
    old rule on post games:  ET iff level & away goals level after 180; pens iff additionally ET goals 0-0
    new rule on pre games :  ET iff level after 180 (exact); pens not identifiable for ties that never played ET."""
    pre, post = s[s.post == 0], s[s.post == 1]
    lvl_aw_eq_post = (post.agg_level_180 == 1) & (post.away_goals_unequal_180 == 0)
    et_post_old = lvl_aw_eq_post.mean()
    pen_post_old = (lvl_aw_eq_post & (post.extra_time_goals == 0)).mean()
    et_pre_new = pre.agg_level_180.mean()
    out = dict(et_pre_obs=pre.extra_time.mean(), et_post_obs=post.extra_time.mean(),
               et_post_under_old_rule=et_post_old, et_pre_under_new_rule=et_pre_new,
               pen_pre_obs=pre.penalties.mean(), pen_post_obs=post.penalties.mean(), pen_post_under_old_rule=pen_post_old)
    tot = out["et_post_obs"] - out["et_pre_obs"]
    mech_a = out["et_post_obs"] - et_post_old           # rule switch evaluated on post-abolition games
    mech_b = et_pre_new - out["et_pre_obs"]             # rule switch evaluated on pre-abolition games
    out.update(et_total=tot, et_mechanical=(mech_a + mech_b) / 2, et_behavioural=tot - (mech_a + mech_b) / 2,
               et_mech_on_post=mech_a, et_mech_on_pre=mech_b,
               pen_total=out["pen_post_obs"] - out["pen_pre_obs"],
               pen_mechanical_on_post=out["pen_post_obs"] - pen_post_old,
               pen_behavioural_on_post=pen_post_old - out["pen_pre_obs"])
    return out


def boot_counterfactual(s, B=B_BOOT):
    pre, post = s[s.post == 0].reset_index(drop=True), s[s.post == 1].reset_index(drop=True)
    keys = ["et_total", "et_mechanical", "et_behavioural", "pen_total", "pen_mechanical_on_post", "pen_behavioural_on_post"]
    draws = {k: [] for k in keys}
    for _ in range(B):
        bs = pd.concat([pre.sample(len(pre), replace=True, random_state=rng.integers(1e9)),
                        post.sample(len(post), replace=True, random_state=rng.integers(1e9))])
        c = counterfactual(bs)
        for k in keys:
            draws[k].append(c[k])
    point = counterfactual(s)
    out = {k: r4(v) for k, v in point.items()}
    for k in keys:
        a = np.array(draws[k])
        out[k + "_ci"] = [r4(np.percentile(a, 2.5)), r4(np.percentile(a, 97.5))]
        out[k + "_p_boot"] = r4(min(1, 2 * min((a <= 0).mean(), (a >= 0).mean())))
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    d = pd.read_csv(CLEAN_CSV)
    d["round"] = d["round"].astype(str)
    d["round_x"] = d["round"].astype(str).replace({"Round of 32": "Entry round", "Knockout play-offs": "Entry round"})
    d["deficit_category"] = pd.Categorical(d.deficit_category, ["1 goal", "level", "2+ goals"])
    d["first_leg_level"] = (d.first_leg_abs_margin == 0).astype(int)
    d["season_c"] = d.season_start - 2018
    d["deficit2p"] = (d.first_leg_abs_margin >= 2).astype(int)
    d["h2pos"] = pd.Categorical(np.select(
        [d.leg2_host_first_leg_margin <= -2, d.leg2_host_first_leg_margin == -1, d.leg2_host_first_leg_margin == 0,
         d.leg2_host_first_leg_margin == 1], ["trail2+", "trail1", "level", "lead1"], "lead2+"),
        ["level", "trail2+", "trail1", "lead1", "lead2+"])
    P = d[d.sample_primary == 1].copy()
    E = d.copy()
    R["samples"] = {"primary": P.groupby("rule_era").size().to_dict(), "extended": E.groupby("rule_era").size().to_dict(),
                    "draft_477": d[d.sample_draft_477 == 1].groupby("rule_era").size().to_dict(),
                    "primary_by_comp_round": P.groupby(["rule_era", "competition", "round"], observed=True).size()
                    .reset_index().rename(columns={0: "n"}).to_dict("records"),
                    "extended_by_round": E.groupby(["rule_era", "round"], observed=True).size().reset_index()
                    .rename(columns={0: "n"}).to_dict("records")}

    # ---- descriptives
    def desc(s):
        g = s.groupby("rule_era")
        t = pd.DataFrame({
            "n_ties": g.size(),
            "leg1_goals_mean": g.leg1_total_goals.mean(), "leg1_goals_sd": g.leg1_total_goals.std(),
            "leg2_goals90_mean": g.leg2_total_goals_90.mean(), "leg2_goals90_sd": g.leg2_total_goals_90.std(),
            "leg2_goals_final_mean": g.leg2_total_goals_final.mean(),
            "tie_goals180_mean": g.tie_total_goals_180.mean(), "away_side_goals180_mean": g.away_side_goals_180.mean(),
            "level_after_180_pct": 100 * g.agg_level_180.mean(),
            "away_goals_would_decide_pct": 100 * g.away_goals_rule_would_decide.mean(),
            "extra_time_pct": 100 * g.extra_time.mean(), "penalties_pct": 100 * g.penalties.mean(),
            "comeback_eligible_n": g.comeback_eligible.sum(), "comeback_pct": 100 * g.comeback.mean(),
            "leg2_host_advanced_pct": 100 * g.leg2_host_advanced.mean()}).T.round(3)
        return t[["pre", "post"]]
    for nm, s in (("primary", P), ("extended", E)):
        t = desc(s); t.to_csv(TAB / f"descriptives_{nm}.csv")
        R[f"descriptives_{nm}"] = t.to_dict()
    res_tab = pd.crosstab([P.rule_era], P.resolution)
    res_tab.to_csv(TAB / "resolution_primary.csv")
    R["resolution_primary"] = res_tab.to_dict("index")
    R["resolution_extended"] = pd.crosstab([E.rule_era], E.resolution).to_dict("index")

    # ---- balance check: first-leg deficit distribution
    ct = pd.crosstab(P.rule_era, P.deficit_category)
    chi = stats.chi2_contingency(ct)
    V = np.sqrt(chi.statistic / (ct.values.sum() * (min(ct.shape) - 1)))
    R["balance_deficit"] = dict(table=ct.to_dict("index"), chi2=r4(chi.statistic), df=int(chi.dof), p=r4(chi.pvalue), cramers_v=r4(V))
    ks = stats.mannwhitneyu(P[P.post == 1].first_leg_abs_margin, P[P.post == 0].first_leg_abs_margin)
    R["balance_margin_mw_p"] = r4(ks.pvalue)
    ct_e = pd.crosstab(E.rule_era, E.deficit_category); chi_e = stats.chi2_contingency(ct_e)
    R["balance_deficit_extended"] = dict(chi2=r4(chi_e.statistic), p=r4(chi_e.pvalue),
                                         cramers_v=r4(np.sqrt(chi_e.statistic / ct_e.values.sum())))

    # ---- unadjusted primary tests
    U = {}
    U["H1_leg2_goals90"] = mean_diff(P, "leg2_total_goals_90")
    U["H2_level_180"] = two_prop(P, "agg_level_180")
    U["H3_extra_time"] = two_prop(P, "extra_time")
    U["H4_penalties"] = two_prop(P, "penalties")
    U["H5_comeback"] = two_prop(P, "comeback")
    U["H6_leg2_host_advanced"] = two_prop(P, "leg2_host_advanced")
    # secondary
    U["S1_leg1_goals"] = mean_diff(P, "leg1_total_goals")
    U["S2_away_side_goals180"] = mean_diff(P, "away_side_goals_180")
    U["S3_leg2_goals_final"] = mean_diff(P, "leg2_total_goals_final")
    U["S4_pens_given_et"] = two_prop(P[P.extra_time == 1], "penalties")
    U["S5_leg2_host_wins_given_et"] = two_prop(P[P.extra_time == 1], "leg2_host_advanced")
    U["S6_away_goals_would_decide"] = two_prop(P, "away_goals_rule_would_decide")
    U["S7_leg1_visitor_goals"] = mean_diff(P, "leg1_away_goals")
    U["S8_leg2_visitor_goals90"] = mean_diff(P, "leg2_away_goals_90")
    U["S10_extra_time_goals_given_et"] = mean_diff(P[P.extra_time == 1], "extra_time_goals")   # exploratory
    # comeback rates within deficit strata (unadjusted)
    for lab, mask in (("1", P.first_leg_abs_margin == 1), ("2plus", P.first_leg_abs_margin >= 2)):
        U[f"S9_comeback_deficit_{lab}"] = two_prop(P[mask], "comeback")
    R["unadjusted_primary"] = U
    # same tests, extended sample
    UE = {"H1_leg2_goals90": mean_diff(E, "leg2_total_goals_90"), "H2_level_180": two_prop(E, "agg_level_180"),
          "H3_extra_time": two_prop(E, "extra_time"), "H4_penalties": two_prop(E, "penalties"),
          "H5_comeback": two_prop(E, "comeback"), "H6_leg2_host_advanced": two_prop(E, "leg2_host_advanced"),
          "S4_pens_given_et": two_prop(E[E.extra_time == 1], "penalties"),
          "S5_leg2_host_wins_given_et": two_prop(E[E.extra_time == 1], "leg2_host_advanced")}
    R["unadjusted_extended"] = UE

    # ---- adjusted models
    M, allrows = {}, []
    # H1 Poisson / NB for leg-2 goals (90 min)
    f_goals = "leg2_total_goals_90 ~ post + C(competition) + C(round, Treatment('Round of 16')) + C(deficit_category)"
    pois = smf.glm(f_goals, P, family=sm.families.Poisson()).fit()
    disp = pois.pearson_chi2 / pois.df_resid
    pois_r = smf.glm(f_goals, P, family=sm.families.Poisson()).fit(cov_type="HC1")
    nb = smf.negativebinomial(f_goals, P).fit(disp=0)
    rows = tidy(pois_r, label="H1 Poisson (robust SE), primary"); allrows += rows
    M["H1_poisson"] = dict(era=era_row(rows), dispersion=r4(disp), nb_alpha=r4(nb.params["alpha"]),
                           nb_alpha_p=r4(nb.pvalues["alpha"]), aic_poisson=r4(pois.aic), aic_nb=r4(nb.aic))
    # H2 level after 180; H3 extra time (both logistic)
    f_tie = "{y} ~ post + C(competition) + C(round, Treatment('Round of 16')) + first_leg_level + deficit2p"
    for key, y in (("H2_level_180", "agg_level_180"), ("H3_extra_time", "extra_time")):
        res, rows = logit(f_tie.format(y=y), P, f"{key} logit, primary"); allrows += rows
        M[key] = dict(era=era_row(rows), n=int(res.nobs))
    # H4 penalties - Firth
    X, y = firth_design("penalties ~ post + C(competition) + first_leg_level", P)
    fr = firth_logit(X.values, y, list(X.columns))
    for x in fr:
        allrows.append(dict(model="H4 penalties Firth logit, primary", term=x["term"], coef=r4(x["coef"]), se=None,
                            ratio=r4(x["OR"]), ci_low=r4(x["ci_low"]), ci_high=r4(x["ci_high"]), p=r4(x["p"])))
    e = [x for x in fr if x["term"] == "post"][0]
    M["H4_penalties_firth"] = dict(era=dict(ratio=r4(e["OR"]), ci=[r4(e["ci_low"]), r4(e["ci_high"])], p=r4(e["p"])))
    # H5 comeback (+ interaction), H6 leg-2 host advancement
    CB = P[P.comeback_eligible == 1].copy()
    f_cb = "comeback ~ post + first_leg_abs_margin + trailer_hosts_leg2 + C(competition) + C(round, Treatment('Round of 16'))"
    res, rows = logit(f_cb, CB, "H5 comeback logit, primary"); allrows += rows
    M["H5_comeback"] = dict(era=era_row(rows), margin=era_row(rows, "first_leg_abs_margin"),
                            trailer_home=era_row(rows, "trailer_hosts_leg2"), n=int(res.nobs))
    f_cbi = "comeback ~ post * deficit2p + trailer_hosts_leg2 + C(competition) + C(round, Treatment('Round of 16'))"
    res, rows = logit(f_cbi, CB, "H5b comeback era x 2+ deficit, primary"); allrows += rows
    M["H5b_comeback_interaction"] = dict(interaction=era_row(rows, "post:deficit2p"), era=era_row(rows), deficit2p=era_row(rows, "deficit2p"))
    f_h = "leg2_host_advanced ~ post + C(h2pos) + C(competition) + C(round, Treatment('Round of 16'))"
    res, rows = logit(f_h, P, "H6 leg-2 host advanced logit, primary"); allrows += rows
    M["H6_leg2_host_advanced"] = dict(era=era_row(rows), n=int(res.nobs))

    # H7 (draft objective 4): does the effect of first-leg deficit on comeback differ by era?
    h7 = {"primary_logit": M["H5b_comeback_interaction"]["interaction"]}
    X, yy = firth_design(f_cbi, CB)
    h7["primary_firth"] = [dict(ratio=r4(x["OR"]), ci=[r4(x["ci_low"]), r4(x["ci_high"])], p=r4(x["p"]))
                           for x in firth_logit(X.values, yy, list(X.columns)) if x["term"] == "post:deficit2p"][0]
    CE = E[E.comeback_eligible == 1].copy()
    f_cbe = "comeback ~ post * deficit2p + trailer_hosts_leg2 + C(competition) + C(round_x)"
    h7["extended_logit"] = era_row(logit(f_cbe, CE, "")[1], "post:deficit2p")
    C19 = CB[CB.covid_disrupted_season == 0]
    h7["primary_excl_2019_20"] = era_row(logit(f_cbi, C19, "")[1], "post:deficit2p")
    h7["primary_season_cluster"] = era_row(logit(f_cbi, CB, "", cluster="season")[1], "post:deficit2p")
    C24 = CB[CB.league_phase_format == 0]
    X, yy = firth_design(f_cbi, C24)   # 0 post-abolition 2+ comebacks before 2024/25 -> separation; Firth needed
    h7["primary_excl_2024_format_firth"] = [dict(ratio=r4(x["OR"]), ci=[r4(x["ci_low"]), r4(x["ci_high"])], p=r4(x["p"]))
                                            for x in firth_logit(X.values, yy, list(X.columns)) if x["term"] == "post:deficit2p"][0]
    # permutation test of the difference-in-differences in raw comeback rates (era labels shuffled within deficit strata)
    def did(df):
        g = df.groupby(["deficit2p", "post"]).comeback.mean()
        return (g[(1, 1)] - g[(1, 0)]) - (g[(0, 1)] - g[(0, 0)])
    obs = did(CB)
    perm = []
    for _ in range(B_PERM):
        sh = CB.copy()
        sh["post"] = sh.groupby("deficit2p").post.transform(lambda s: rng.permutation(s.values))
        perm.append(did(sh))
    perm = np.array(perm)
    h7["did_pp"] = r4(100 * obs)
    h7["did_perm_p"] = r4((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B_PERM + 1))
    M["H7_deficit_interaction"] = h7
    R["models_primary"] = M
    pd.DataFrame(allrows).to_csv(TAB / "regression_models_primary.csv", index=False)

    # ---- Holm correction across the seven primary hypotheses
    prim = {"H1_leg2_goals90": U["H1_leg2_goals90"]["p_welch"], "H2_level_180": U["H2_level_180"]["p_z"],
            "H3_extra_time": U["H3_extra_time"]["p_z"], "H4_penalties": U["H4_penalties"]["p_fisher"],
            "H5_comeback": U["H5_comeback"]["p_z"], "H6_leg2_host_advanced": U["H6_leg2_host_advanced"]["p_z"],
            "H7_deficit_interaction": h7["primary_logit"]["p"]}
    holm = multipletests(list(prim.values()), method="holm")[1]
    R["holm"] = {k: dict(p=r4(p), p_holm=r4(ph)) for (k, p), ph in zip(prim.items(), holm)}
    for k in prim:
        if k in U:
            U[k]["p_primary_test"] = R["holm"][k]["p"]; U[k]["p_holm"] = R["holm"][k]["p_holm"]

    # ---- counterfactual: who would have advanced under the away-goals rule? (post-abolition ties)
    def old_rule_winner(r):
        a1 = r.leg1_home_goals + r.leg2_away_goals_90; a2 = r.leg1_away_goals + r.leg2_home_goals_90
        if a1 != a2:
            return r.leg1_home_team if a1 > a2 else r.leg2_home_team
        if r.leg2_away_goals_90 != r.leg1_away_goals:
            return r.leg1_home_team if r.leg2_away_goals_90 > r.leg1_away_goals else r.leg2_home_team
        eh = r.leg2_home_goals_final - r.leg2_home_goals_90; ea = r.leg2_away_goals_final - r.leg2_away_goals_90
        if eh != ea:
            return r.winner
        return r.leg1_home_team if ea > 0 else r.winner   # level ET with goals: visitor wins on away goals
    cfw = {}
    for nm, s in (("primary", P), ("extended", E)):
        p = s[s.post == 1].copy()
        p["winner_old_rule"] = p.apply(old_rule_winner, axis=1)
        p["comeback_old_rule"] = np.where(p.comeback_eligible == 1, (p.winner_old_rule == p.first_leg_trailer).astype(float), np.nan)
        ch = p[p.winner_old_rule != p.winner]
        cfw[nm] = dict(n_post=len(p), n_changed=len(ch), share_changed=r4(len(ch) / len(p)),
                       changed=ch[["tie_id", "season", "competition", "round", "leg2_home_team", "leg1_home_team",
                                   "leg1_home_goals", "leg1_away_goals", "leg2_home_goals_90", "leg2_away_goals_90",
                                   "resolution", "winner", "winner_old_rule"]].to_dict("records"),
                       comeback_obs={k: r4(p[m].comeback.mean()) for k, m in (("all", p.comeback_eligible == 1), ("1", p.first_leg_abs_margin == 1), ("2plus", p.first_leg_abs_margin >= 2))},
                       comeback_old_rule={k: r4(p[m].comeback_old_rule.mean()) for k, m in (("all", p.comeback_eligible == 1), ("1", p.first_leg_abs_margin == 1), ("2plus", p.first_leg_abs_margin >= 2))},
                       leg2_host_adv_obs=r4(p.leg2_host_advanced.mean()),
                       leg2_host_adv_old_rule=r4((p.winner_old_rule == p.leg2_home_team).mean()))
        if nm == "extended":
            ch.to_csv(TAB / "counterfactual_changed_winners_extended.csv", index=False)
    R["counterfactual_winners"] = cfw
    # pre-abolition comebacks that were completed via the away-goals rule
    pc = P[(P.post == 0) & (P.comeback == 1)]
    R["pre_comebacks_via_away_goals"] = {lab: dict(n=int(len(pc[m])), via_away_goals=int((pc[m].resolution == "away goals (180 min)").sum()))
                                         for lab, m in (("1", pc.first_leg_abs_margin == 1), ("2plus", pc.first_leg_abs_margin >= 2))}

    # ---- counterfactual decomposition
    R["decomposition_primary"] = boot_counterfactual(P)
    R["decomposition_extended"] = boot_counterfactual(E, B=4000)

    # ---- comeback rates by deficit size x era (for figure)
    cb = (P[P.comeback_eligible == 1].assign(m=lambda x: x.first_leg_abs_margin.clip(upper=3))
          .groupby(["rule_era", "m"]).comeback.agg(["sum", "count"]).reset_index())
    cb["rate"] = cb["sum"] / cb["count"]
    cb[["lo", "hi"]] = [proportion_confint(a, b, method="wilson") for a, b in zip(cb["sum"], cb["count"])]
    cb.to_csv(TAB / "comeback_by_margin_primary.csv", index=False)
    R["comeback_by_margin"] = cb.round(4).to_dict("records")

    # ---- robustness grid: era effect for each outcome across specifications
    specs = {}
    base_ctrl = "+ C(competition) + C(round_x) + first_leg_level + deficit2p"
    specs["Primary sample (R16-SF)"] = (P, base_ctrl, None)
    specs["Extended sample (all 533 ties)"] = (E, base_ctrl, None)
    specs["Draft sample (477 ties)"] = (d[d.sample_draft_477 == 1], base_ctrl, None)
    specs["Primary, excluding 2019/20"] = (P[P.covid_disrupted_season == 0], base_ctrl, None)
    specs["Primary, excluding 2024+ format"] = (P[P.league_phase_format == 0], base_ctrl, None)
    specs["Primary, UCL only"] = (P[P.competition == "UCL"], "+ C(round_x) + first_leg_level + deficit2p", None)
    specs["Primary, UEL only"] = (P[P.competition == "UEL"], "+ C(round_x) + first_leg_level + deficit2p", None)
    specs["Primary, + linear season trend"] = (P, base_ctrl + " + season_c", None)
    specs["Primary, season-clustered SE"] = (P, base_ctrl, "season")
    outcomes = {"Second-leg goals (90 min)": ("leg2_total_goals_90", "poisson"), "Level after 180 min": ("agg_level_180", "logit"),
                "Extra time": ("extra_time", "logit"), "Penalty shoot-out": ("penalties", "firth"),
                "Second-leg host advanced": ("leg2_host_advanced", "logit")}
    grid = []
    for sname, (s, ctrl, clus) in specs.items():
        for oname, (y, kind) in outcomes.items():
            f = f"{y} ~ post {ctrl}"
            try:
                if kind == "poisson":
                    m = smf.glm(f, s, family=sm.families.Poisson())
                    res = m.fit(cov_type="cluster", cov_kwds={"groups": s[clus]}) if clus else m.fit(cov_type="HC1")
                    rr = era_row(tidy(res))
                elif kind == "logit":
                    res, rows = logit(f, s, "", cluster=clus); rr = era_row(rows)
                else:
                    X, yy = firth_design(f"{y} ~ post + C(competition) + first_leg_level" if "only" not in sname
                                         else f"{y} ~ post + first_leg_level", s)
                    if "trend" in sname:
                        X, yy = firth_design(f"{y} ~ post + C(competition) + first_leg_level + season_c", s)
                    e = [x for x in firth_logit(X.values, yy, list(X.columns)) if x["term"] == "post"][0]
                    rr = dict(ratio=r4(e["OR"]), ci=[r4(e["ci_low"]), r4(e["ci_high"])], p=r4(e["p"]))
            except Exception as ex:  # pragma: no cover
                rr = dict(ratio=None, ci=[None, None], p=None, error=str(ex))
            if rr.get("error"):
                print("robustness error:", sname, oname, rr["error"])
            raw_rate = s.groupby("post")[y].mean()
            grid.append(dict(spec=sname, outcome=oname, n_pre=int((s.post == 0).sum()), n_post=int((s.post == 1).sum()),
                             pre=r4(raw_rate.get(0)), post=r4(raw_rate.get(1)), ratio=rr["ratio"],
                             ci_low=rr["ci"][0], ci_high=rr["ci"][1], p=rr["p"]))
    G = pd.DataFrame(grid); G.to_csv(TAB / "robustness_grid.csv", index=False)
    R["robustness_grid"] = grid

    # ---- placebo tests (no rule change): split the pre era at 2016/17; split the post era at 2024/25
    pl = []
    for nm, s, cut in (("Placebo: pre era, 2016/17+ vs 2012/13-2015/16", P[P.post == 0], 2016),
                       ("Placebo: post era, 2024/25+ vs 2021/22-2023/24", P[P.post == 1], 2024)):
        s = s.assign(fake=(s.season_start >= cut).astype(int))
        for oname, col in (("Level after 180 min", "agg_level_180"), ("Extra time", "extra_time"), ("Penalty shoot-out", "penalties")):
            t = two_prop(s.assign(post=s.fake), col)
            pl.append(dict(test=nm, outcome=oname, early=t["pre_p"], late=t["post_p"], n_early=t["pre_n"], n_late=t["post_n"],
                           diff_pp=t["diff_pp"], p_fisher=t["p_fisher"]))
        t = mean_diff(s.assign(post=s.fake), "leg2_total_goals_90")
        pl.append(dict(test=nm, outcome="Second-leg goals (90 min)", early=t["pre_mean"], late=t["post_mean"],
                       n_early=t["pre_n"], n_late=t["post_n"], diff_pp=t["diff"], p_fisher=t["p_welch"]))
    pd.DataFrame(pl).to_csv(TAB / "placebo_tests.csv", index=False)
    R["placebo"] = pl

    # ---- power / minimum detectable effects (80% power, two-sided alpha .05, primary sample sizes)
    n_pre, n_post = int((P.post == 0).sum()), int((P.post == 1).sum())
    ratio = n_post / n_pre
    mde = {}
    dmin = TTestIndPower().solve_power(nobs1=n_post, ratio=1 / ratio, alpha=0.05, power=0.8)
    sd = P.leg2_total_goals_90.std()
    mde["leg2_goals90_goals"] = r4(dmin * sd); mde["leg2_goals90_d"] = r4(dmin)
    for k, col in (("extra_time", "extra_time"), ("level_180", "agg_level_180"), ("penalties", "penalties")):
        p0 = P[P.post == 0][col].mean()
        hmin = NormalIndPower().solve_power(nobs1=n_post, ratio=1 / ratio, alpha=0.05, power=0.8)
        p1 = np.sin(np.arcsin(np.sqrt(p0)) + hmin / 2) ** 2
        mde[k + "_baseline"] = r4(p0); mde[k + "_detectable_post_rate"] = r4(p1); mde[k + "_mde_pp"] = r4(100 * (p1 - p0))
    # achieved power for observed ET / penalty differences
    for k, col in (("extra_time", "extra_time"), ("penalties", "penalties")):
        h = proportion_effectsize(P[P.post == 1][col].mean(), P[P.post == 0][col].mean())
        mde[k + "_power_observed_effect"] = r4(NormalIndPower().power(effect_size=h, nobs1=n_post, ratio=1 / ratio, alpha=0.05))
    R["power"] = mde

    # ---- season-level series
    ss = (E.groupby(["season", "rule_era"]).agg(n=("tie_id", "size"), et=("extra_time", "mean"), pens=("penalties", "mean"),
                                               level=("agg_level_180", "mean"), goals2=("leg2_total_goals_90", "mean"))
          .reset_index())
    sp = (P.groupby(["season", "rule_era"]).agg(n=("tie_id", "size"), et=("extra_time", "mean"), pens=("penalties", "mean"),
                                               level=("agg_level_180", "mean"), goals2=("leg2_total_goals_90", "mean"),
                                               awaydec=("away_goals_rule_would_decide", "mean"))
          .reset_index())
    sp.to_csv(TAB / "season_series_primary.csv", index=False); ss.to_csv(TAB / "season_series_extended.csv", index=False)
    R["season_series_primary"] = sp.round(4).to_dict("records")

    # ---- goal distribution comparison (chi-square on 0..6+ goals)
    g = pd.crosstab(P.rule_era, P.leg2_total_goals_90.clip(upper=6))
    chi_g = stats.chi2_contingency(g)
    R["goal_distribution"] = dict(table=g.to_dict("index"), chi2=r4(chi_g.statistic), df=int(chi_g.dof), p=r4(chi_g.pvalue))
    ks2 = stats.ks_2samp(P[P.post == 1].leg2_total_goals_90, P[P.post == 0].leg2_total_goals_90)
    R["goal_distribution"]["ks_p"] = r4(ks2.pvalue)

    with open(RES / "results.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_markdown(R)
    print("analysis complete")


def write_markdown(R):
    U = R["unadjusted_primary"]; M = R["models_primary"]; D = R["decomposition_primary"]; H = R["holm"]
    L = ["# Analysis output (generated by src/analysis.py)", "",
         f"Primary sample: {R['samples']['primary']}; extended sample: {R['samples']['extended']}.", "",
         "## Primary hypotheses (primary sample, Holm-adjusted across H1-H7)", "",
         "| Hypothesis | Pre | Post | Difference (95% CI) | p | Holm p | Adjusted ratio (95% CI) | p (model) |",
         "|---|---|---|---|---|---|---|---|"]
    mk = {"H1_leg2_goals90": "H1_poisson", "H2_level_180": "H2_level_180", "H3_extra_time": "H3_extra_time",
          "H4_penalties": "H4_penalties_firth", "H5_comeback": "H5_comeback", "H6_leg2_host_advanced": "H6_leg2_host_advanced"}
    for k, mkey in mk.items():
        u = U[k]; m = M[mkey]["era"]
        if "pre_mean" in u:
            a, b = f"{u['pre_mean']:.2f} (SD {u['pre_sd']:.2f})", f"{u['post_mean']:.2f} (SD {u['post_sd']:.2f})"
            c = f"{u['diff']:+.2f} ({u['diff_ci'][0]:.2f}, {u['diff_ci'][1]:.2f})"
        else:
            a = f"{u['pre_x']}/{u['pre_n']} ({100*u['pre_p']:.1f}%)"; b = f"{u['post_x']}/{u['post_n']} ({100*u['post_p']:.1f}%)"
            c = f"{u['diff_pp']:+.1f} pp ({u['diff_ci_pp'][0]:.1f}, {u['diff_ci_pp'][1]:.1f})"
        L.append(f"| {k} | {a} | {b} | {c} | {H[k]['p']:.4f} | {H[k]['p_holm']:.4f} | "
                 f"{m['ratio']:.2f} ({m['ci'][0]:.2f}, {m['ci'][1]:.2f}) | {m['p']:.4f} |")
    h7 = M["H7_deficit_interaction"]["primary_logit"]
    L.append(f"| H7_deficit_interaction | | | DiD {M['H7_deficit_interaction']['did_pp']:+.1f} pp | {H['H7_deficit_interaction']['p']:.4f} | "
             f"{H['H7_deficit_interaction']['p_holm']:.4f} | {h7['ratio']:.3f} ({h7['ci'][0]:.3f}, {h7['ci'][1]:.3f}) | {h7['p']:.4f} |")
    L += ["", "## H7 sensitivity", "", "```", json.dumps(M["H7_deficit_interaction"], indent=1), "```",
          "", "## Mechanical vs behavioural decomposition", "", "Primary:", "```", json.dumps(D, indent=1), "```",
          "Extended:", "```", json.dumps(R["decomposition_extended"], indent=1), "```",
          "", "## Counterfactual winners under the away-goals rule (post-abolition ties)", "", "```",
          json.dumps({k: {x: v[x] for x in v if x != "changed"} for k, v in R["counterfactual_winners"].items()}, indent=1), "```",
          "", "## Power (80%, alpha .05, primary sample sizes)", "", "```", json.dumps(R["power"], indent=1), "```"]
    (RES / "analysis_output.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
