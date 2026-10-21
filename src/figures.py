"""Step 4 - Publication figures (PNG 300 dpi + PDF) from data/processed and results/results.json."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from statsmodels.stats.proportion import proportion_confint
from common import CLEAN_CSV, FIG, RES, TAB

PRE, POST = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (CVD-safe pair)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal blue ramp (steps 250/400/550/700)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": SURF})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


def ygrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, ha="right", va="bottom")


# ------------------------------------------------------------------ Figure 1: design + rule logic
def fig1(d, R):
    fig = plt.figure(figsize=(7.0, 4.3))
    ax = fig.add_axes([0, 0, 0.47, 1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    def box(x, y, w, h, txt, fc="#f4f4f2", ec=MUTED, bold=False, fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else None, linespacing=1.25)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=8))
    box(0.5, 8.5, 9, 1.2, "533 two-legged UCL + UEL knockout ties\n2012/13-2019/20 and 2021/22-2025/26", bold=True)
    box(0.5, 6.35, 9, 1.4, "Excluded before data entry: finals (single match), 2020/21 season\n(empty stadiums), 2019/20 single-leg ties played in Aug 2020,\nLeipzig v Spartak 2021/22 (walkover). Every leg verified\nagainst openfootball / Wikipedia: 1,066 / 1,066 match", fs=6.5)
    arrow(5, 8.5, 5, 7.75)
    box(0.3, 3.3, 4.4, 2.3, "PRIMARY sample\nRound of 16, QF, SF\n(same rounds, both eras)\n\npre 210  |  post 139", fc="#e8f0fb", ec=PRE, bold=True, fs=7.0)
    box(5.3, 3.3, 4.4, 2.3, "EXTENDED sample\nall rounds, incl. UEL\nR32 / knockout play-offs\n\npre 338  |  post 195", fc="#fdeee6", ec=POST, bold=True, fs=7.0)
    arrow(3, 6.35, 2.5, 5.6); arrow(7, 6.35, 7.5, 5.6)
    box(0.3, 0.3, 9.4, 2.4, "Analysis: Welch / exact / permutation tests, TOST equivalence,\nPoisson & logistic models (Firth for rare events), Holm\ncorrection for 7 primary hypotheses, counterfactual re-scoring\nof every tie under the other era's tie-break rule, 9\nrobustness specifications, 2 placebo splits", fs=6.9)
    arrow(2.5, 3.3, 3.5, 2.7); arrow(7.5, 3.3, 6.5, 2.7)
    ax.text(0.2, 9.95, "A", fontsize=11, fontweight="bold", va="top")

    ax2 = fig.add_axes([0.5, 0, 0.5, 1]); ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.text(0.2, 9.95, "B", fontsize=11, fontweight="bold", va="top")
    def b2(x, y, w, h, txt, fc, ec, fs=7.2, bold=False):
        ax2.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec=ec, lw=0.8))
        ax2.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2, fontweight="bold" if bold else None)
    def a2(x1, y1, x2, y2, lab=None, side="left"):
        ax2.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=8))
        if lab:
            ax2.text((x1 + x2) / 2 + (-0.15 if side == "left" else 0.15), (y1 + y2) / 2, lab, fontsize=6.6, color=INK2,
                     ha="right" if side == "left" else "left", va="center")
    b2(2.2, 8.8, 5.6, 0.9, "Aggregate score after 180 min", "#f4f4f2", MUTED, bold=True)
    b2(0.1, 6.9, 3.0, 0.9, "Leader advances", "#f4f4f2", MUTED)
    a2(3.4, 8.8, 1.8, 7.8, "unequal")
    b2(4.0, 6.9, 2.0, 0.9, "Level", "#f4f4f2", MUTED, bold=True)
    a2(5.0, 8.8, 5.0, 7.8)
    b2(6.9, 6.75, 3.0, 1.2, "Compare\naway goals", "#e8f0fb", PRE)
    a2(6.0, 7.35, 6.9, 7.35)
    ax2.text(6.45, 7.62, "pre", fontsize=6.6, color=PRE, ha="center", fontweight="bold")
    b2(6.9, 4.7, 3.0, 1.2, "Away-goals\nwinner advances", "#e8f0fb", PRE, fs=6.9)
    a2(8.4, 6.75, 8.4, 5.9, "unequal", side="right")
    b2(2.6, 4.4, 3.4, 1.5, "Extra time, 30 min,\nat second-leg host\n(pre: ET away goals\nstill count)", "#f4f4f2", MUTED, fs=6.6)
    a2(5.0, 6.9, 4.3, 5.9)
    ax2.text(4.55, 6.4, "post", fontsize=6.6, color=POST, ha="right", fontweight="bold")
    a2(7.4, 6.75, 5.9, 5.9)
    ax2.text(6.75, 6.15, "equal", fontsize=6.6, color=INK2, ha="left")
    b2(2.6, 2.2, 3.4, 1.0, "Penalty shoot-out", "#fdeee6", POST)
    a2(4.3, 4.4, 4.3, 3.2, "still level")
    b2(0.1, 0.1, 9.8, 1.5, "Re-scoring each era's ties under the other rule (primary sample):\nabolition alone adds +8.4 pp extra time and +5.8 pp shoot-outs", "#fdeee6", POST, fs=6.9)
    save(fig, "fig1_design_and_rules")


# ------------------------------------------------------------------ Figure 2: resolution pathways
def fig2(d):
    cats = ["aggregate (180 min)", "away goals (180 min)", "extra time", "penalty shoot-out"]
    labels = ["Aggregate in 180 min", "Away goals (180 min)", "Decided in extra time", "Penalty shoot-out"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (nm, s), letter in zip(axes, (("Primary sample (R16-SF)", d[d.sample_primary == 1]), ("Extended sample (all rounds)", d)), "AB"):
        y = [1, 0]
        for yi, era in zip(y, ("pre", "post")):
            sub = s[s.rule_era == era]
            share = sub.resolution.value_counts(normalize=True).reindex(cats).fillna(0).values * 100
            left = 0
            for k, (v, c) in enumerate(zip(share, RAMP)):
                if v > 0:
                    ax.barh(yi, v - 0.4 if v > 1 else v, left=left + (0.2 if v > 1 else 0), height=0.56, color=c)
                    if k == 0:
                        ax.text(64, yi, f"{v:.1f}", ha="left", va="center", fontsize=7.2, color=INK)
                    if k > 0 and v >= 2.5:
                        ax.text(left + v / 2, yi, f"{v:.1f}", ha="center", va="center", fontsize=7.2, color="white" if k >= 1 else INK)
                left += v
            ax.text(101, yi, f"n={len(sub)}", va="center", fontsize=7.2, color=INK2)
        ax.set_yticks(y, ["Away goals\nrule active", "Abolished"])
        ax.set_xlim(60, 108); ax.set_xticks([60, 70, 80, 90, 100], ["60%", "70", "80", "90", "100%"])
        ax.set_xlabel("Cumulative share of ties (axis starts at 60%)", fontsize=7.5)
        ax.set_title(nm, fontsize=9); panel(ax, letter)
        ax.xaxis.grid(True); ax.set_axisbelow(True)
    handles = [Patch(color=c, label=l) for c, l in zip(RAMP, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=7.6)
    fig.tight_layout(); save(fig, "fig2_resolution_pathways")


# ------------------------------------------------------------------ Figure 3: decomposition
def fig3(R):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, key, letter in zip(axes, ("primary", "extended"), "AB"):
        D = R[f"decomposition_{key}"]
        groups = [("Extra time", D["et_pre_obs"], D["et_post_under_old_rule"], D["et_post_obs"]),
                  ("Penalty shoot-out", D["pen_pre_obs"], D["pen_post_under_old_rule"], D["pen_post_obs"])]
        for g, (name, pre_obs, post_old, post_obs) in enumerate(groups):
            x0 = g * 3.0
            for i, v in enumerate([pre_obs, post_old, post_obs]):
                kw = dict(color="white", edgecolor=POST, hatch="////", lw=1.0) if i == 1 else dict(color=[PRE, None, POST][i])
                ax.bar(x0 + i * 0.8, 100 * v, width=0.68, **kw)
                ax.text(x0 + i * 0.8, 100 * v + 0.4, f"{100*v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks([0.8, 3.8], ["Extra time", "Penalty shoot-out"], fontsize=8)
        ax.set_ylim(0, 22); ygrid(ax)
        ax.set_title("Primary sample (R16-SF)" if key == "primary" else "Extended sample (all rounds)", fontsize=9)
        ax.text(0.98, 0.97, f"Extra-time change {100*D['et_total']:+.1f} pp\n= mechanical {100*D['et_mechanical']:+.1f}\n+ behavioural {100*D['et_behavioural']:+.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2, linespacing=1.3)
        panel(ax, letter)
    axes[0].set_ylabel("% of ties")
    handles = [Patch(color=PRE, label="Rule active, observed"),
               Patch(facecolor="white", edgecolor=POST, hatch="////", label="Post-abolition ties re-scored under away-goals rule"),
               Patch(color=POST, label="Abolished, observed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=7.4)
    fig.tight_layout(); save(fig, "fig4_mechanical_decomposition")


# ------------------------------------------------------------------ Figure 4: goals
def fig4(d, R):
    P = d[d.sample_primary == 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ks = np.arange(0, 7)
    for off, era, c in ((-0.19, "pre", PRE), (0.19, "post", POST)):
        v = P[P.rule_era == era].leg2_total_goals_90.clip(upper=6).value_counts(normalize=True).reindex(ks).fillna(0) * 100
        ax.bar(ks + off, v.values, width=0.36, color=c, label=("Away goals rule active" if era == "pre" else "Abolished") + f" (n={int((P.rule_era == era).sum())})")
    ax.set_xticks(ks, [str(k) for k in ks[:-1]] + ["6+"]); ax.set_xlabel("Goals in second leg (90 min)"); ax.set_ylabel("% of ties")
    ax.set_ylim(0, 33); ygrid(ax); ax.legend(loc="upper right", fontsize=7.2)
    g = R["goal_distribution"]; ax.set_title(f"Distribution: chi2({g['df']}) = {g['chi2']:.2f}, p = {g['p']:.2f}", fontsize=8.5); panel(ax, "A")
    ax = axes[1]
    U = R["unadjusted_primary"]
    rows = [("First leg, total", U["S1_leg1_goals"]), ("Second leg, 90 min", U["H1_leg2_goals90"]),
            ("Second leg incl. ET", U["S3_leg2_goals_final"]), ("Visiting teams, 180 min", U["S2_away_side_goals180"]),
            ("Leg-1 visitor", U["S7_leg1_visitor_goals"]), ("Leg-2 visitor, 90 min", U["S8_leg2_visitor_goals90"])]
    ax.axvspan(-0.5, 0.5, color="#f1f0ec", zorder=0)
    ax.axvline(0, color=MUTED, lw=0.7)
    for i, (lab, u) in enumerate(rows[::-1]):
        ax.plot(u["diff_ci"], [i, i], color=INK2, lw=1.4, solid_capstyle="round")
        ax.plot(u["diff"], i, "o", ms=6, color=POST, mec="white", mew=1.5)
        ax.text(0.66, i, f"TOST p = {u['p_tost']:.3f}" if u['p_tost'] >= 0.001 else "TOST p < 0.001", ha="left", va="center", fontsize=6.6, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7.4)
    ax.set_xlim(-0.6, 1.25); ax.set_xticks([-0.5, 0, 0.5]); ax.set_xlabel("Post - pre difference in mean goals (95% CI)")
    ax.set_ylim(-0.9, len(rows) - 0.5)
    ax.text(0, -0.75, "equivalence margin +/-0.5 goals", ha="center", fontsize=6.4, color=INK2)
    ax.set_title("Mean differences and equivalence tests", fontsize=8.5); panel(ax, "B")
    fig.tight_layout(); save(fig, "fig3_goals")


# ------------------------------------------------------------------ Figure 5: comebacks by deficit
def fig5(d, R):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    P = d[(d.sample_primary == 1) & (d.comeback_eligible == 1)].copy()
    P["m"] = np.where(P.first_leg_abs_margin >= 2, "2+ goals", "1 goal")
    for off, era, c, lab in ((-0.12, "pre", PRE, "Away goals rule active"), (0.12, "post", POST, "Abolished")):
        xs, ys, lo, hi = [], [], [], []
        for k, mm in enumerate(["1 goal", "2+ goals"]):
            s = P[(P.rule_era == era) & (P.m == mm)]
            x, n = int(s.comeback.sum()), len(s)
            l, h = proportion_confint(x, n, method="wilson")
            xs.append(k + off); ys.append(100 * x / n); lo.append(100 * l); hi.append(100 * h)
            ax.text(k + off + 0.05, 100 * x / n, f" {x}/{n}", fontsize=6.8, color=INK2, va="center")
        ax.vlines(xs, lo, hi, color=c, lw=1.6)
        ax.plot(xs, ys, "o", color=c, ms=7, mec="white", mew=1.5, label=lab)
        ax.plot(xs, ys, color=c, lw=1, alpha=0.5)
    ax.set_xticks([0, 1], ["1-goal deficit", "2+-goal deficit"]); ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("Comeback rate, % (95% Wilson CI)"); ygrid(ax); ax.legend(loc="upper right", fontsize=7.4)
    h7 = R["models_primary"]["H7_deficit_interaction"]
    ax.set_title(f"Era x deficit: OR {h7['primary_logit']['ratio']:.2f}, Holm p = {R['holm']['H7_deficit_interaction']['p_holm']:.3f}", fontsize=8.5)
    panel(ax, "A")
    ax = axes[1]
    specs = [("Primary logit", h7["primary_logit"]), ("Primary Firth", h7["primary_firth"]), ("Extended sample", h7["extended_logit"]),
             ("Excl. 2019/20", h7["primary_excl_2019_20"]), ("Season-clustered SE", h7["primary_season_cluster"]),
             ("Excl. 2024+ (Firth)", h7["primary_excl_2024_format_firth"])]
    ax.axvline(1, color=MUTED, lw=0.7)
    for i, (lab, e) in enumerate(specs[::-1]):
        ax.plot(e["ci"], [i, i], color=INK2, lw=1.4)
        ax.plot(e["ratio"], i, "D", color=POST, ms=5.5, mec="white", mew=1.2)
        ax.text(1.02, i, f"p={e['p']:.3f}", transform=ax.get_yaxis_transform(), fontsize=6.6, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_xlim(0.0003, 2); ax.set_xticks([0.001, 0.01, 0.1, 1], ["0.001", "0.01", "0.1", "1"])
    ax.set_yticks(range(len(specs)), [s[0] for s in specs[::-1]], fontsize=7.4)
    ax.set_xlabel("Interaction odds ratio (log scale)"); ax.set_title("Robustness of the interaction", fontsize=8.5); panel(ax, "B")
    fig.tight_layout(); save(fig, "fig6_comebacks")


# ------------------------------------------------------------------ Figure 6: forest across specs
def fig6(R):
    G = pd.DataFrame(R["robustness_grid"])
    outs = ["Extra time", "Penalty shoot-out", "Level after 180 min", "Second-leg host advanced"]
    specs = list(dict.fromkeys(G.spec))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.3), sharey=True)
    for ax, o, letter in zip(axes, outs, "ABCD"):
        s = G[G.outcome == o].set_index("spec").reindex(specs)
        ax.axvline(1, color=MUTED, lw=0.7)
        for i, sp in enumerate(specs[::-1]):
            r = s.loc[sp]
            sig = r.p < 0.05
            ax.plot([max(r.ci_low, 0.08), min(r.ci_high, 30)], [i, i], color=INK2, lw=1.2)
            ax.plot(r.ratio, i, "o", color=POST if sig else "white", mec=POST, mew=1.3, ms=5.5)
        ax.set_xscale("log"); ax.set_xlim(0.08, 30); ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
        ax.set_title(o, fontsize=8); ax.xaxis.grid(True); ax.set_axisbelow(True)
        panel(ax, letter)
    axes[0].set_yticks(range(len(specs)), specs[::-1], fontsize=7)
    fig.supxlabel("Adjusted odds ratio, abolished vs active (95% CI, log scale); filled = p < 0.05", fontsize=8, color=INK2)
    fig.tight_layout(); save(fig, "fig7_robustness_forest")


# ------------------------------------------------------------------ Figure 7: season series
def fig7(d):
    P = d[d.sample_primary == 1]
    g = P.groupby("season").agg(level=("agg_level_180", "mean"), et=("extra_time", "mean"), n=("tie_id", "size"),
                                 era=("post", "first"), start=("season_start", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = g.start.values
    ax.axvspan(2020.5, 2025.6, color="#fdeee6", zorder=0)
    ax.text(2020.65, 47, "away goals abolished", fontsize=7, color=POST, va="top")
    ax.text(2012, 47, "away goals rule active", fontsize=7, color=PRE, va="top")
    for era in (0, 1):
        m = g.era.values == era
        ax.plot(x[m], 100 * g.level[m], color=INK2, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="Aggregate level after 180 min" if era == 0 else None)
        ax.plot(x[m], 100 * g.et[m], color=POST, lw=2, marker="s", ms=5, mec="white", mew=1.2, label="Extra time played" if era == 0 else None)
    for era in (0, 1):
        s = P[P.post == era]
        ax.hlines(100 * s.agg_level_180.mean(), x[g.era == era].min() - 0.3, x[g.era == era].max() + 0.3, color=INK2, lw=0.8, linestyles="solid", alpha=0.5)
    ax.set_xticks(x, [s.replace("/", "/\n") for s in g.season], fontsize=7)
    ax.set_ylabel("% of ties (primary sample)"); ax.set_ylim(0, 48); ygrid(ax)
    ax.legend(loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, -0.22))
    ax.plot([], [], color=INK2, lw=0.8, alpha=0.5, label="Era mean, level after 180 min"); ax.legend(loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, -0.22))
    ax.text(2020.0, 1.5, "2020/21\nexcluded", fontsize=6.3, color=MUTED, ha="center")
    ax.annotate("after abolition every level tie goes to\nextra time, so the two series coincide", xy=(2022, 10.8), xytext=(2021.0, 35),
                fontsize=6.6, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    fig.tight_layout(); save(fig, "fig5_season_series")


def main():
    d = pd.read_csv(CLEAN_CSV)
    R = json.load(open(RES / "results.json"))
    fig1(d, R); fig2(d); fig3(R); fig4(d, R); fig5(d, R); fig6(R); fig7(d)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
