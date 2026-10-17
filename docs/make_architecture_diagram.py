"""Draws docs/architecture.png (run from the repo root: python docs/make_architecture_diagram.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
CODE, LLM, HUMAN, DATA = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
W, H = 17.2, 9.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, sub, color, filled=False, dashed=False, size=7.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       fc=color if filled else SURF, ec=color, lw=2, ls="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, weight="bold",
            color="white" if filled else INK)
    ax.text(x + w / 2, y + h - 0.55, sub, ha="center", va="top", fontsize=size,
            color="white" if filled else INK2, linespacing=1.35)
    return (x, y, w, h)


def arrow(a, b, side_a="r", side_b="l", color=INK2, dashed=False, label=None, rad=0.0, lx=0, ly=0):
    def pt(bx, side):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]
    p1, p2 = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.5, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, fontsize=7.5, color=color, ha="center")


ax.text(0.2, 8.95, "Agentic claims investigation: one orchestrator, six specialised tools", fontsize=15,
        weight="bold", color=INK)
ax.text(0.2, 8.58, "src/agent/graph.py  ·  docs/agent_restructure_plan.md  ·  every recommendation is reviewed by a "
        "human adjuster; the system never denies a claim", fontsize=9.5, color=INK2)

y, h, w, g = 5.55, 1.75, 1.62, 0.24
xs = [0.2 + i * (w + g) for i in range(9)]
claim = box(xs[0], y, w, h, "Claim", "CSV / JSON /\nform dict", DATA)
ing = box(xs[1], y, w, h, "ingest", "normalise to\nClaimSchema;\ninvestigation id", CODE)
val = box(xs[2], y, w, h, "validate", "13 rules\n(blocking /\nwarning / info)", CODE)
sco = box(xs[3], y, w, h, "Tool 1 · score", "probability,\npredicted class,\nrisk band, red flags", CODE)
orc = box(xs[4], y, w, h, "Orchestrator", "LLM ReAct loop\n≤ max_steps;\nplans tool calls", LLM, filled=True)
cmp_ = box(xs[5], y, w, h, "complete", "runs required\ntools it skipped\n(logged 'auto')", CODE)
agg = box(xs[6], y, w, h, "aggregate", "statuses, missing\ninfo, conflicts,\ncompleteness", CODE)
syn = box(xs[7], y, w, h, "reasoning", "LLM JSON:\ndecision, rationale,\ncited evidence", LLM, filled=True)
grd = box(xs[8], y, w, h, "guardrails", "recommendation\n+ reason codes\n(code)", CODE, filled=True)
for a, b in zip((claim, ing, val, sco, orc, cmp_, agg, syn), (ing, val, sco, orc, cmp_, agg, syn, grd)):
    arrow(a, b)

tools = box(xs[4] - 0.75, 2.45, 3.1, 2.3, "Tools 2–6 (claim-bound)",
            "2  explain_fraud_score · SHAP\n3  retrieve_policy\n4  verify_policy · PASS/FAIL/UNKNOWN\n"
            "5  find_similar_claims\n6  get_claim_history\nprerequisites run first (dependency)", CODE, size=7.6)
arrow(orc, tools, "b", "t", color=LLM, label="tool calls", lx=-0.5, rad=0.2)
arrow(tools, orc, "t", "b", color=CODE, label="results", lx=0.55, rad=0.2)
arrow(tools, cmp_, "r", "b", color=MUTED, dashed=True, rad=0.25)

dx = 4.55
model = box(xs[3] - 0.2, 7.45, 2.0, 0.95, "models/", "calibrated LightGBM · SHAP\nexplainer · model card", DATA, size=7.2)
faiss = box(dx, 3.75, 1.9, 1.0, "FAISS index", "10,793 real training\nclaims; never synthetic", DATA, size=7.2)
pol = box(dx, 2.35, 1.9, 1.2, "policy store", "data/policies/*.json;\nelse claim-recorded\nattributes", DATA, size=7.0)
hist = box(dx, 0.95, 1.9, 1.2, "history source", "claim_history.source;\nnone here →\n'unavailable'", DATA, size=7.0)
arrow(model, sco, "b", "t", color=DATA)
for b in (faiss, pol, hist):
    arrow(tools, b, "l", "r", color=DATA, rad=0.0)

human = box(xs[8], 2.95, w, 1.75, "human review", "LangGraph interrupt;\nadjuster approves,\nmodifies or overrides", HUMAN, filled=True)
audit = box(xs[8], 1.0, w, 1.45, "audit log", "trace: every node,\ntool call + origin,\nguardrail, decision", DATA)
fb = box(xs[7], 2.95, w, 1.75, "rule-based\nfallback", "\n\nLLM failed twice\nor no LLM", CODE, dashed=True)
arrow(grd, human, "b", "t", color=HUMAN)
arrow(human, audit, "b", "t", color=DATA)
arrow(syn, fb, "b", "t", color=MUTED, dashed=True, label="invalid ×2", lx=0.5)
ax.add_patch(FancyArrowPatch((xs[7] + w, 4.45), (xs[8] + 0.25, y), arrowstyle="-|>", mutation_scale=13, lw=1.5,
                             color=MUTED, ls="--", shrinkA=2, shrinkB=2))

ax.text(0.2, 4.75, "Guardrails (code, not prompt)", fontsize=10, weight="bold", color=INK)
for i, t in enumerate(["3 decisions only; never deny",
                       "blocking validation → more info",
                       "high band never APPROVE",
                       "APPROVE needs score, no policy FAIL",
                       "untraceable evidence removed",
                       "LLM fails twice → rule-based decision",
                       "reason codes, statuses: from code",
                       "human review on every decision"]):
    ax.text(0.25, 4.38 - i * 0.39, "•  " + t, fontsize=8, color=INK2)

for i, (c, lab, filled) in enumerate([(CODE, "deterministic code", False), (LLM, "LLM step", True),
                                       (HUMAN, "human", True), (DATA, "data / artifacts", False)]):
    ax.add_patch(FancyBboxPatch((0.3 + i * 2.05, 0.3), 0.3, 0.22, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc=c if filled else SURF, ec=c, lw=1.6))
    ax.text(0.7 + i * 2.05, 0.41, lab, fontsize=8.5, va="center", color=INK2)
fig.savefig("docs/architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
print("ok")
