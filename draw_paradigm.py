"""
draw_paradigm.py — Generate schematic figure for the octagon sequence task

Creates a 3-panel illustration showing the three sequence types
(Simple, Nested, Random) on the regular octagon with annotated transitions.

Output: results/paradigm_schematic.png

Author: Puzhi YU
Date:   January 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ── Global style ──────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.facecolor': 'white',
})

N = 8
angles = [2 * np.pi * k / N + np.pi/2 for k in range(N)]  # start from top
R = 2.0  # octagon radius
cx = [R * np.cos(a) for a in angles]
cy = [R * np.sin(a) for a in angles]

# Colors
VERTEX_COLOR = '#2C3E50'
SIMPLE_COLOR = '#E74C3C'
NESTED_FWD   = '#3498DB'
NESTED_BWD   = '#F39C12'
RANDOM_COLOR = '#95A5A6'
BG_OCTAGON   = '#ECF0F1'


def draw_octagon(ax, alpha=1.0):
    """Draw the base octagon with vertex labels."""
    # Fill
    octagon = plt.Polygon(list(zip(cx, cy)), closed=True,
                           fc=BG_OCTAGON, ec='#BDC3C7', lw=1.5, alpha=0.5)
    ax.add_patch(octagon)
    # Edges
    for i in range(N):
        j = (i + 1) % N
        ax.plot([cx[i], cx[j]], [cy[i], cy[j]], '-', color='#BDC3C7', lw=1, alpha=0.6)
    # Vertices
    for k in range(N):
        circle = plt.Circle((cx[k], cy[k]), 0.22, fc='white', ec=VERTEX_COLOR, lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(cx[k], cy[k], str(k), ha='center', va='center',
                fontsize=12, fontweight='bold', color=VERTEX_COLOR, zorder=6)


def curved_arrow(ax, start_k, end_k, color, lw=2.0, rad=0.25, offset=0.28, style='->', zorder=4):
    """Draw a curved arrow between two vertices."""
    x0, y0 = cx[start_k], cy[start_k]
    x1, y1 = cx[end_k], cy[end_k]
    # offset from center of vertex circle
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/dist, dy/dist
    x0 += ux * offset
    y0 += uy * offset
    x1 -= ux * offset
    y1 -= uy * offset
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='->', mutation_scale=14,
        connectionstyle=f'arc3,rad={rad}',
        color=color, lw=lw, zorder=zorder,
        path_effects=[pe.withStroke(linewidth=lw+1.5, foreground='white')]
    )
    ax.add_patch(arrow)


# ══════════════════════════════════════════════════
# Create figure: 3 panels
# ══════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 6.5))

# ── Panel (a): Simple sequence ────────────────────
ax1 = fig.add_subplot(131)
draw_octagon(ax1)
# Show path: 0→1→2→3→4→5 (clockwise +1)
simple_path = [0, 1, 2, 3, 4, 5]
for i in range(len(simple_path) - 1):
    curved_arrow(ax1, simple_path[i], simple_path[i+1], SIMPLE_COLOR, lw=2.5, rad=0.15)
# Highlight start
ax1.add_patch(plt.Circle((cx[0], cy[0]), 0.30, fc=SIMPLE_COLOR, alpha=0.15, ec='none', zorder=3))

ax1.set_xlim(-3.2, 3.2)
ax1.set_ylim(-3.2, 3.2)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('(a)  Simple Rule\n$x_{t+1} = (x_t + 1)\\,\\mathrm{mod}\\,8$',
              fontsize=13, fontweight='bold', pad=12)
# Sequence annotation
ax1.text(0, -2.85, '0 → 1 → 2 → 3 → 4 → 5 → ...',
         ha='center', va='center', fontsize=11, color=SIMPLE_COLOR,
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=SIMPLE_COLOR, alpha=0.9))

# ── Panel (b): Nested sequence ────────────────────
ax2 = fig.add_subplot(132)
draw_octagon(ax2)
# Show path: 0→2→1→3→2→4 (alternating +2, -1)
nested_path = [0, 2, 1, 3, 2, 4]
for i in range(len(nested_path) - 1):
    step = nested_path[i+1] - nested_path[i]
    if step > 0 or step == -6:  # forward (+2)
        c = NESTED_FWD
        rad = 0.2
    else:  # backward (-1)
        c = NESTED_BWD
        rad = -0.2
    curved_arrow(ax2, nested_path[i], nested_path[i+1], c, lw=2.5, rad=rad)

ax2.add_patch(plt.Circle((cx[0], cy[0]), 0.30, fc=NESTED_FWD, alpha=0.15, ec='none', zorder=3))

ax2.set_xlim(-3.2, 3.2)
ax2.set_ylim(-3.2, 3.2)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('(b)  Nested Rule\nEven: $+2$,  Odd: $-1$  (mod 8)',
              fontsize=13, fontweight='bold', pad=12)
ax2.text(0, -2.85, '0 → 2 → 1 → 3 → 2 → 4 → ...',
         ha='center', va='center', fontsize=11, color=NESTED_FWD,
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=NESTED_FWD, alpha=0.9))
# Legend for nested
fwd_patch = mpatches.Patch(color=NESTED_FWD, label='+2 (forward)')
bwd_patch = mpatches.Patch(color=NESTED_BWD, label='−1 (backward)')
ax2.legend(handles=[fwd_patch, bwd_patch], loc='lower left', fontsize=9,
           framealpha=0.9, edgecolor='#BDC3C7')

# ── Panel (c): Random sequence ────────────────────
ax3 = fig.add_subplot(133)
draw_octagon(ax3)
# Random jumps: 0→5→2→7→1
random_path = [0, 5, 2, 7, 1]
for i in range(len(random_path) - 1):
    curved_arrow(ax3, random_path[i], random_path[i+1], RANDOM_COLOR, lw=2.0,
                 rad=0.25 * (1 if i % 2 == 0 else -1))
ax3.add_patch(plt.Circle((cx[0], cy[0]), 0.30, fc=RANDOM_COLOR, alpha=0.15, ec='none', zorder=3))

# Question marks for unpredictability
for k in [3, 4, 6]:
    ax3.text(cx[k]*1.25, cy[k]*1.25, '?', ha='center', va='center',
             fontsize=16, color=RANDOM_COLOR, fontweight='bold', alpha=0.5)

ax3.set_xlim(-3.2, 3.2)
ax3.set_ylim(-3.2, 3.2)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('(c)  Random (Control)\n$x_{t+1} \\sim \\mathcal{U}\\{0,\\ldots,7\\}$',
              fontsize=13, fontweight='bold', pad=12)
ax3.text(0, -2.85, '0 → 5 → 2 → 7 → 1 → ...',
         ha='center', va='center', fontsize=11, color='#7F8C8D',
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RANDOM_COLOR, alpha=0.9))

# ── Global title ──────────────────────────────────
fig.suptitle('Octagon Geometric Sequence Task — Three Difficulty Levels',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
save_path = 'results/paradigm_schematic.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Paradigm schematic saved: {save_path}")
