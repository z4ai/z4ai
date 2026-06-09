# SPDX-License-Identifier: Apache-2.0
"""Render the z4ai pipeline diagram (paper/pipeline.png) for the paper.

Regenerate with:  python paper/make_pipeline_figure.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#1b2430"
EXP = "#2f6f4f"  # low-entropy plane (green)
MAN = "#8a5a2b"  # noise-like plane (amber)
CORE = "#274b8f"  # codec core (blue)
OUT = "#5a3d7a"  # container (purple)

fig, ax = plt.subplots(figsize=(7.2, 6.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")


def box(x, y, w, h, text, color, fc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=color,
            facecolor=fc,
            mutation_aspect=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
        wrap=True,
    )
    return (x + w / 2, y, x + w / 2, y + h)  # bottom-center, top-center helpers


def arrow(x1, y1, x2, y2, color=INK):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


# Input
box(3.0, 10.7, 4.0, 0.95, "Float tensor bytes\n(bf16 / fp16 / fp32 / fp64)", INK)
# Split
box(
    3.0, 9.0, 4.0, 0.95, "Plane / bit-field split\n[ sign | exponent | mantissa ]", CORE
)
# Two planes
box(0.5, 7.0, 4.0, 1.1, "Exponent / sign plane\n(low entropy)\n→ rANS or zstd", EXP)
box(5.5, 7.0, 4.0, 1.1, "Mantissa plane\n(noise-like)\n→ store or zstd", MAN)
# Whole-tensor matching
box(
    2.5,
    5.0,
    5.0,
    1.05,
    "Whole-tensor long-distance matching\n(dedup tied / repeated weights)",
    CORE,
)
# Best-of
box(
    2.5,
    3.2,
    5.0,
    1.05,
    "Best-of selection\n(keep smallest; never worse than zstd)",
    CORE,
)
# Output container
box(
    2.0,
    1.2,
    6.0,
    1.1,
    "Self-describing container (.z4ai / .zstn)\nper-tensor index → random-access reads",
    OUT,
    fc="#f3eefa",
)

# Arrows
arrow(5.0, 10.7, 5.0, 9.95)  # input -> split
arrow(4.2, 9.0, 2.5, 8.1)  # split -> exponent
arrow(5.8, 9.0, 7.5, 8.1)  # split -> mantissa
arrow(2.5, 7.0, 4.2, 6.05)  # exponent -> matching
arrow(7.5, 7.0, 5.8, 6.05)  # mantissa -> matching
arrow(5.0, 5.0, 5.0, 4.25)  # matching -> best-of
arrow(5.0, 3.2, 5.0, 2.3)  # best-of -> container

ax.text(
    5.0,
    0.5,
    "Decoding is the exact inverse, driven entirely by the header.",
    ha="center",
    va="center",
    fontsize=8.5,
    style="italic",
    color="#55606e",
)

plt.tight_layout()
fig.savefig("paper/pipeline.png", dpi=200, bbox_inches="tight")
print("wrote paper/pipeline.png")
