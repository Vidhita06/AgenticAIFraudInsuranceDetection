"""Draws docs/architecture.png (run from the repo root: python docs/make_architecture_diagram.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
CODE, LLM, HUMAN, DATA = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 15); ax.set_ylim(0, 8.6); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, title, sub, color, filled=False, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       fc=color if filled else SURF, ec=color, lw=2, ls="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - (0.18 if h < 1 else 0.28), title, ha="center", va="top", fontsize=10.5, weight="bold",
            color="white" if filled else INK)
    ax.text(x + w / 2, y + h - (0.55 if h < 1 else 0.62), sub, ha="center", va="top", fontsize=8, color="white" if filled else INK2,
            linespacing=1.35)
    return (x, y, w, h)

def arrow(a, b, side_a="r", side_b="l", color=INK2, dashed=False, label=None, rad=0.0, lx=0, ly=0):
    def pt(bx, side):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]
    p1, p2 = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, fontsize=7.5, color=color, ha="center")

ax.text(0.2, 8.35, "Agentic claims triage: LangGraph architecture", fontsize=15, weight="bold", color=INK)
ax.text(0.2, 7.98, "src/agent/graph.py  ·  every recommendation is reviewed by a human adjuster; the system never denies a claim",
        fontsize=9.5, color=INK2)

y = 5.2; h = 1.55; w = 1.75
intake = box(0.2, y, 1.5, h, "Claim", "CSV / JSON /\nform dict", DATA)
ing = box(2.05, y, w, h, "1 ingest", "parse + normalise\nto ClaimSchema;\nlist missing fields", CODE)
val = box(4.15, y, w, h, "2 validate", "13 rules from\npolicy_rules.yaml\n(blocking / warning)", CODE)
sco = box(6.25, y, w, h, "3 score", "calibrated fraud\nprobability, risk band,\ntop SHAP factors", CODE)
agt = box(8.35, y, 2.0, h, "4 agent (LLM)", "ReAct loop, ≤ MAX_STEPS\nchooses which tools\nto call", LLM, filled=True)
syn = box(10.75, y, 1.9, h, "5 synthesize", "LLM writes JSON:\ndecision, rationale,\nevidence (cited)", LLM, filled=True)
grd = box(13.0, y, 1.85, h, "6 guardrails", "enforced in code\n(see below)", CODE, filled=True)

tools = box(8.15, 2.55, 2.4, 1.75, "tools", "find_similar_claims(k)\ncheck_policy_rules()\n→ ToolMessages", CODE)
faiss = box(5.75, 2.6, 2.05, 1.6, "FAISS index", "10,793 real training\nclaims only (cosine)\ndata/vector_store/", DATA)
rules = box(5.75, 0.65, 2.05, 1.6, "policy_rules.yaml", "validation rules +\n12 red flags with\nfraud rate & lift", DATA)
model = box(4.6, 6.98, 3.2, 0.92, "models/", "preprocessor · calibrated model\nSHAP explainer · model card", DATA)
human = box(12.75, 2.55, 2.1, 1.75, "7 human review", "LangGraph interrupt;\nadjuster approves,\nmodifies or overrides", HUMAN, filled=True)
audit = box(12.75, 0.65, 2.1, 1.35, "audit log", "trace of every node,\ntool call, guardrail\n+ adjuster decision", DATA)
fb = box(10.85, 2.55, 1.6, 1.75, "rule-based\nfallback", "\n\nLLM failed twice\nor no LLM", CODE, dashed=True)

for a, b in ((intake, ing), (ing, val), (val, sco), (sco, agt), (agt, syn), (syn, grd)):
    arrow(a, b)
arrow(agt, tools, "b", "t", color=LLM, label="tool calls", lx=-0.55, rad=0.25)
arrow(tools, agt, "t", "b", color=CODE, label="results", lx=0.62, rad=0.25)
arrow(tools, faiss, "l", "r")
arrow(tools, rules, "b", "r", rad=-0.3)
arrow(model, sco, "b", "t", color=DATA)
arrow(syn, fb, "b", "t", color=MUTED, dashed=True, label="invalid ×2", lx=0.55, rad=0.0)
ax.add_patch(FancyArrowPatch((12.45, 3.9), (13.25, 5.2), arrowstyle="-|>", mutation_scale=14, lw=1.6,
                             color=MUTED, ls="--", connectionstyle="arc3,rad=0.25", shrinkA=2, shrinkB=2))
ax.add_patch(FancyArrowPatch((14.2, 5.2), (14.2, 4.3), arrowstyle="-|>", mutation_scale=14, lw=1.6,
                             color=HUMAN, shrinkA=2, shrinkB=2))
arrow(human, audit, "b", "t", color=DATA)

ax.text(0.2, 3.95, "Guardrails (code, not prompt)", fontsize=10, weight="bold", color=INK)
for i, t in enumerate(["only APPROVE / FLAG / REQUEST_MORE_INFO; never deny",
                       "blocking validation failure → REQUEST_MORE_INFO",
                       "high risk band can never be APPROVE",
                       "evidence not traceable to a tool output is removed + logged",
                       "LLM fails or invalid output twice → rule-based decision",
                       "requires_human_review = true on every decision"]):
    ax.text(0.3, 3.55 - i * 0.4, "•  " + t, fontsize=8.6, color=INK2)

for i, (c, lab, filled) in enumerate([(CODE, "deterministic code", False), (LLM, "LLM step", True),
                                       (HUMAN, "human", True), (DATA, "data / artifacts", False)]):
    ax.add_patch(FancyBboxPatch((0.3 + i * 2.05, 0.25), 0.3, 0.22, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc=c if filled else SURF, ec=c, lw=1.6))
    ax.text(0.7 + i * 2.05, 0.36, lab, fontsize=8.5, va="center", color=INK2)
fig.savefig("docs/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
print("ok")
