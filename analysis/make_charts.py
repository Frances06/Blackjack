"""Turn the results CSVs into report-ready charts (PNG, 200 dpi).

Run from the repo root, after the experiments have produced their CSVs:
    python3 -m machine_learning.train_classifier
    python3 -m simulation.compare_agents
    python3 -m analysis.make_charts

Charts are written to analysis/figures/. Any chart whose CSV is missing is
skipped with a message, so you can run this at any point.

  agent_edges.png         edge of every agent, flat vs count betting, 95% CIs
  classifier_comparison   CV accuracy and EV regret for each classifier
  tree_depth.png          decision-tree accuracy vs max_depth
  mc_learning_curve.png   RL agent's edge as training progresses
  mc_settings.png         RL learning curves for the three settings tried
  strategy_mc_vs_table    the RL agent's learned strategy, with cells that
                          differ from the dataset's best action outlined
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "figures")
SIM = os.path.join(ROOT, "simulation", "results")
ML = os.path.join(ROOT, "machine_learning", "results")

# Colour-blind-checked categorical palette (fixed order) and chart ink.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9,
})


def read(path):
    if not os.path.isfile(path):
        print(f"skipped: {os.path.relpath(path, ROOT)} not found (run the experiment first)")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def pct(x):
    return f"{x:+.2f}%"


# ---------------------------------------------------------------- charts
def agent_edges():
    rows = read(os.path.join(SIM, "agent_comparison.csv"))
    if not rows:
        return
    rows = rows[::-1]                                   # first agent at the top
    names = [r["agent"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(rows) + 1.4))
    ax.axvline(0, color=INK2, linewidth=1)
    ax.text(0, -0.55, " break-even", color=INK2, fontsize=8, va="bottom")
    for off, col, key, label in ((0.14, BLUE, "flat", "Flat 10-chip bets"),
                                 (-0.14, ORANGE, "count", "Hi-Lo count betting")):
        for i, r in enumerate(rows):
            e, se = 100 * float(r[f"{key}_edge"]), 100 * 1.96 * float(r[f"{key}_edge_se"])
            ax.plot([e - se, e + se], [i + off] * 2, color=col, linewidth=2, solid_capstyle="round")
            ax.plot(e, i + off, "o", color=col, markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label if i == 0 else None)
    ax.set_yticks(range(len(rows)), names)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Edge: profit per chip of initial bet (%), with 95% confidence interval")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("Player edge by agent", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    save(fig, "agent_edges.png")


def classifier_comparison():
    rows = read(os.path.join(ML, "classifier_comparison.csv"))
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["cv_accuracy"]))
    names = [r["model"] for r in rows]
    acc = [100 * float(r["cv_accuracy"]) for r in rows]
    reg = [float(r["cv_regret_cents"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    a1.barh(names, acc, color=BLUE, height=0.6)
    for y, v in enumerate(acc):
        a1.text(v - 1.5, y, f"{v:.1f}%", va="center", ha="right", color="white", fontsize=9)
    a1.set_xlim(0, 100)
    a1.set_xlabel("Accuracy (%)")
    a1.set_title("Cross-validated accuracy")
    a2.barh(names, reg, color=BLUE, height=0.6)
    a2.set_xscale("log")
    for y, v in enumerate(reg):
        a2.text(v * 1.15, y, f"{v:g}", va="center", ha="left", color=INK2, fontsize=9)
    a2.set_xlim(min(reg) / 2, max(reg) * 6)
    a2.set_xlabel("EV lost per decision (cents per $, log scale) - lower is better")
    a2.set_title("Cost of mistakes")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
        a.tick_params(axis="y", length=0)
    save(fig, "classifier_comparison.png")


def tree_depth():
    rows = read(os.path.join(ML, "tree_depth.csv"))
    if not rows:
        return
    labels = [r["max_depth"] for r in rows]
    acc = [100 * float(r["cv_accuracy"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(range(len(rows)), acc, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    best = max(range(len(acc)), key=lambda i: (acc[i], -i))
    ax.annotate(f"{acc[best]:.1f}% from depth {labels[best]}", (best, acc[best]),
                xytext=(0, -22), textcoords="offset points", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("CV accuracy (%)")
    ax.set_ylim(min(acc) - 5, 102)
    ax.set_title("Decision tree: accuracy vs depth")
    save(fig, "tree_depth.png")


def mc_learning_curve():
    rows = read(os.path.join(ML, "mc_learning_curve.csv"))
    if not rows:
        return
    x = [int(r["episodes"]) / 1e6 for r in rows]
    y = [100 * float(r["greedy_edge"]) for r in rows]
    se = [100 * 1.96 * float(r["edge_se"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_between(x, [a - b for a, b in zip(y, se)], [a + b for a, b in zip(y, se)],
                    color=BLUE, alpha=0.15, linewidth=0, label="95% confidence interval")
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Monte Carlo agent (greedy policy)")
    agents = read(os.path.join(SIM, "agent_comparison.csv")) or []
    table = next((r for r in agents if r["agent"].startswith("Lookup")), None)
    if table:
        t = 100 * float(table["flat_edge"])
        ax.axhline(t, color=INK2, linewidth=1)
        ax.text(x[-1], t, f"lookup table {pct(t)}", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.set_xlabel("Training hands (millions)")
    ax.set_ylabel("Edge, flat bets (%)")
    ax.set_title("Reinforcement learning: edge during training")
    ax.legend(loc="lower right")
    save(fig, "mc_learning_curve.png")


def mc_settings():
    rows = read(os.path.join(ML, "mc_settings_comparison.csv"))
    if not rows:
        return
    cols = [c for c in rows[0] if c.startswith("greedy_edge_")]
    x = [int(c.split("_")[-1].rstrip("M")) for c in cols]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for r, col in zip(rows, (BLUE, ORANGE, AQUA)):
        y = [100 * float(r[c]) for c in cols]
        label = f"alpha_min {r['alpha_min']}, epsilon {r['eps_start']}->{r['eps_end']}"
        ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.text(x[-1] + 0.25, y[-1], pct(y[-1]), va="center", color=INK2, fontsize=8)
    ax.set_xlim(x[0] - 0.5, x[-1] + 1.6)
    ax.set_xticks(x)
    ax.set_xlabel("Training hands (millions)")
    ax.set_ylabel("Edge, flat bets (%)")
    ax.set_title("Reinforcement learning: effect of learning settings")
    ax.legend(loc="lower right")
    save(fig, "mc_settings.png")


def strategy_mc_vs_table():
    """Strategy chart of the RL agent's first decision, vs the dataset's best action."""
    try:
        from machine_learning.classifier import load_rows
        from machine_learning.monte_carlo import MonteCarloAgent
        from simulation.compare_agents import _cards_for
        from simulation.blackjack import Card
        rows = [r for r in load_rows() if int(r["decks"]) == 6 and r["dealer_rule"] == "H17"]
        agent = MonteCarloAgent.load()
    except FileNotFoundError as e:
        print(f"skipped: strategy chart ({e})")
        return
    letters = {"STAND": "S", "HIT": "H", "DOUBLE": "D", "SPLIT": "P"}
    colours = {"H": BLUE, "S": ORANGE, "D": AQUA, "P": YELLOW}
    text_on = {"H": "white", "S": "white", "D": "white", "P": INK}
    ups = [str(i) for i in range(2, 11)] + ["A"]
    sections = [
        ("Hard totals", "hard", [str(t) for t in range(5, 20)]),
        ("Soft totals", "soft", [f"A,{k}" for k in range(2, 10)]),
        ("Pairs", "pair", [f"{v},{v}" for v in range(2, 11)] + ["A,A"]),
    ]
    index = {(r["category"], r["hand"], r["dealer_upcard"]): r["best_action"] for r in rows}
    heights = [len(h) for _, _, h in sections]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), gridspec_kw={"wspace": 0.18})
    diff = total = 0
    for ax, (title, cat, hands) in zip(axes, sections):
        for yi, hand in enumerate(hands):
            for xi, up in enumerate(ups):
                row = {"category": cat, "hand": hand, "dealer_upcard": up}
                cards = _cards_for(row)
                mc = letters[agent(cards, Card(up), can_double=True, can_split=cat == "pair")]
                best = index.get((cat, hand, up))
                ax.add_patch(Rectangle((xi + 0.04, yi + 0.04), 0.92, 0.92, color=colours[mc], linewidth=0))
                ax.text(xi + 0.5, yi + 0.5, mc, ha="center", va="center", fontsize=8.5,
                        color=text_on[mc], fontweight="bold")
                if best and best != mc:
                    diff += 1
                    ax.add_patch(Rectangle((xi + 0.06, yi + 0.06), 0.88, 0.88, fill=False,
                                           edgecolor=INK, linewidth=2.2))
                    ax.text(xi + 0.9, yi + 0.13, best, ha="right", va="top", fontsize=6,
                            color=text_on[mc])
                total += best is not None
        ax.set_xlim(0, len(ups))
        ax.set_ylim(len(hands), 0)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_xticks([i + 0.5 for i in range(len(ups))], ups)
        ax.set_yticks([i + 0.5 for i in range(len(hands))], hands)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, pad=8, loc="left")
        ax.set_xlabel("Dealer upcard", fontsize=9)
        ax.xaxis.set_label_position("bottom")
    fig.suptitle(f"Strategy learned by the Monte Carlo agent (6 decks, H17): "
                 f"{total - diff} of {total} first decisions match the dataset",
                 x=0.125, ha="left", fontsize=12, fontweight="bold", y=1.0)
    handles = [Patch(color=colours[k], label=n) for k, n in
               (("H", "H  Hit"), ("S", "S  Stand"), ("D", "D  Double"), ("P", "P  Split"))]
    handles.append(Patch(facecolor="none", edgecolor=INK, linewidth=2,
                         label="Differs from dataset (small letter = dataset's choice)"))
    fig.legend(handles=handles, loc="upper left", ncol=5, bbox_to_anchor=(0.12, 0.08))
    save(fig, "strategy_mc_vs_table.png")


def main():
    for chart in (agent_edges, classifier_comparison, tree_depth,
                  mc_learning_curve, mc_settings, strategy_mc_vs_table):
        chart()


if __name__ == "__main__":
    main()
