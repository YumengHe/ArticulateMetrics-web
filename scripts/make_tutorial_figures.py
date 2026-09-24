#!/usr/bin/env python3
"""Draw the tutorial figures of docs.html, one figure per concept.

Every number in these figures is computed with the articulatearena library
(installed in the conda env), so the pictures stay consistent with the code:
E values, norms, the compactification, and the tree distances are the real
ones, not sketches. Styling follows the ablation figures of the paper (serif
text, the SPARK part palette, white ground).

Usage:
    python scripts/make_tutorial_figures.py     # writes static/tutorial/*.png
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

from articulatearena.new_equation.continuous import (  # noqa: E402
    compactified_joint_metric,
    continuous_joint_repr,
    radial_compactify,
)
from articulatearena.new_equation.E import joint_metric  # noqa: E402
from articulatearena.new_equation.inner_product import (  # noqa: E402
    make_kinetic_energy_norm,
    make_split_norm,
)
from articulatearena.new_equation.representation import (  # noqa: E402
    joint_to_endpoint_pair,
    make_endpoint_pair,
    screw_twist,
    wrap_revolute_limits,
)
from articulatearena.new_equation.tree import (  # noqa: E402
    assignment_tree_distance,
    exact_tree_distance,
)
from articulatearena.types import Joint, LinkInertia  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "static" / "tutorial"

# SPARK part palette + site ink tones (same as the ablation figures).
BLUE, AMBER, TEAL, BRICK = "#5B8DB8", "#E5B455", "#6FB5AE", "#C0564F"
SAGE, MAUVE, GREY = "#92B573", "#C58BC0", "#B9BEC4"
INK, INK2, HAIR = "#1f1e1b", "#55524b", "#e3dfd6"

plt.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
DPI = 200

SPLIT = make_split_norm(1.0)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


def _clean(ax, grid: bool = True) -> None:
    for s in ax.spines.values():
        s.set_color(INK)
    if grid:
        ax.grid(True, color=HAIR, linewidth=0.7)
        ax.set_axisbelow(True)


def _E(pair1, pair2, ip=SPLIT) -> float:
    return float(joint_metric(pair1.u.data, pair1.v.data, pair2.u.data, pair2.v.data, ip))


# ---------------------------------------------------------------------------
# 1. A joint as a screw twist: xi = (omega, v) for the three finite types
# ---------------------------------------------------------------------------
def fig_twist_embeddings() -> None:
    fig = plt.figure(figsize=(12, 4.2))
    a = np.array([0.0, 0.0, 1.0])
    o = np.array([0.9, 0.5, 0.0])
    panels = [
        ("revolute", 0.0, r"$\xi_{\rm rev} = (\mathbf{a},\ \mathbf{o}\times\mathbf{a})$"),
        ("prismatic", 0.0, r"$\xi_{\rm pris} = (0,\ \mathbf{a})$"),
        ("helical", 0.35, r"$\xi_{\rm hel} = (\mathbf{a},\ \mathbf{o}\times\mathbf{a} + h\,\mathbf{a})$"),
    ]
    for i, (jtype, pitch, formula) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        xi = screw_twist(a, o, jtype, pitch).data
        omega, v = xi[:3], xi[3:]
        # world frame
        for d, lab in ((np.array([1, 0, 0]), "x"), (np.array([0, 1, 0]), "y"), (np.array([0, 0, 1]), "z")):
            ax.quiver(0, 0, 0, *(0.45 * d), color=GREY, arrow_length_ratio=0.15, linewidth=1)
            ax.text(*(0.52 * d), lab, color=INK2, fontsize=11)
        # axis line through o (revolute/helical) or the direction only (prismatic)
        if jtype != "prismatic":
            ax.plot([o[0], o[0]], [o[1], o[1]], [-0.2, 1.3], color=INK, linestyle=":", linewidth=1)
            ax.scatter(*o, color=INK, s=18)
            ax.text(o[0] + 0.05, o[1] + 0.05, -0.05, r"$\mathbf{o}$", color=INK)
            ax.quiver(*o, *(0.8 * a), color=BLUE, arrow_length_ratio=0.12, linewidth=2.2)
            ax.text(o[0] + 0.05, o[1] + 0.05, 0.85, r"$\mathbf{a}$", color=BLUE)
            # moment vector at the world origin
            ax.quiver(0, 0, 0, *v, color=BRICK, arrow_length_ratio=0.12, linewidth=2.2)
            ax.text(v[0] * 1.15, v[1] * 1.15 - 0.05, v[2] * 1.15 - (0.22 if pitch else 0.0), r"$\mathbf{v} = \mathbf{o}\times\mathbf{a}$" + (r"$\,+h\mathbf{a}$" if pitch else ""), color=BRICK, fontsize=11)
            # rotation glyph
            t = np.linspace(0.2, 1.6 * math.pi, 40)
            ax.plot(o[0] + 0.3 * np.cos(t), o[1] + 0.3 * np.sin(t), 0.55 + (0.12 * t / (2 * math.pi) if pitch else 0 * t), color=AMBER, linewidth=2)
        else:
            ax.quiver(*o, *(0.8 * a), color=BLUE, arrow_length_ratio=0.12, linewidth=2.2)
            ax.text(o[0] + 0.05, o[1] + 0.05, 0.85, r"$\mathbf{a}$", color=BLUE)
            ax.scatter(*o, color=GREY, s=18)
            ax.text(o[0] - 0.55, o[1] + 0.05, -0.12, r"$\mathbf{o}$ (irrelevant)", color=INK2, fontsize=10)
            ax.quiver(0, 0, 0, *v, color=BRICK, arrow_length_ratio=0.12, linewidth=2.2)
            ax.text(v[0] + 0.05, v[1] + 0.05, v[2] + 0.05, r"$\mathbf{v} = \mathbf{a}$", color=BRICK, fontsize=11)
            # sliding glyph: a small block along the axis
            ax.plot([o[0]] * 2, [o[1]] * 2, [0.2, 0.5], color=AMBER, linewidth=6, solid_capstyle="butt")
        ax.set_title(f"{jtype}\n{formula}", fontsize=12)
        ax.set_xlim(-0.2, 1.3); ax.set_ylim(-0.2, 1.3); ax.set_zlim(-0.2, 1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1, 1, 1))
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white"); pane.set_edgecolor(HAIR)
        ax.grid(False)
    fig.text(0.5, -0.02, r"$\omega$ = rotational part (blue axis), $\mathbf{v}$ = translational part (red); the moment $\mathbf{o}\times\mathbf{a}$ places the axis without naming a point on it",
             ha="center", color=INK2, fontsize=11)
    _save(fig, "twist_embeddings.png")


# ---------------------------------------------------------------------------
# 2. The endpoint pair: a joint is a segment through the origin of se(3)
# ---------------------------------------------------------------------------
def _dir(deg: float) -> np.ndarray:
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def fig_endpoint_pair() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    xi1, l1 = _dir(22), (0.0, 1.5)
    xi2, l2 = _dir(58), (-0.5, 1.2)
    for xi, (lo, hi), color, name in ((xi1, l1, AMBER, "1"), (xi2, l2, BLUE, "2")):
        p, q = lo * xi, hi * xi
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=5, solid_capstyle="round", zorder=3)
        ax.scatter([p[0], q[0]], [p[1], q[1]], color=color, s=70, zorder=4, edgecolor="white")
        ax.annotate("", xy=xi * 0.55, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)
        perp = np.array([-xi[1], xi[0]])
        ax.text(*(xi * 0.62 + perp * 0.16), rf"$\xi_{name}$", color=color, fontsize=13)
        ax.text(*(p + np.array([0.04, -0.14])), rf"$z_{name}^- = l_{name}^-\,\xi_{name}$", color=color, fontsize=12)
        ax.text(*(q + np.array([0.04, 0.05])), rf"$z_{name}^+ = l_{name}^+\,\xi_{name}$", color=color, fontsize=12)
    ax.scatter([0], [0], color=INK, s=60, zorder=5)
    ax.text(0.05, -0.28, r"$J_0 = \{0, 0\}$ (fixed joint)", color=INK, fontsize=12)
    # cost to fixed for joint 1
    ax.annotate("", xy=(l1[1] * xi1) * 0.98, xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1, linestyle="--", shrinkA=6, shrinkB=6), zorder=1)
    pair1 = make_endpoint_pair(np.r_[xi1, 0, 0, 0, 0], *l1)
    zero = make_endpoint_pair(np.r_[xi1, 0, 0, 0, 0], 0.0, 0.0)
    ax.text(0.95, 0.05, rf"$E(J_1, J_0) = {_E(pair1, zero):.2f}$" + "\n(the motion the joint carries)", color=INK2, fontsize=11, ha="center")
    ax.set_xlim(-0.6, 1.9); ax.set_ylim(-0.75, 1.55)
    ax.set_aspect("equal")
    ax.set_xlabel(r"twist space $\mathfrak{se}(3)$, a 2-D slice"); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("The endpoint pair: a joint is a segment through the origin")
    _clean(ax, grid=False)
    _save(fig, "endpoint_pair.png")


# ---------------------------------------------------------------------------
# 3. The quotient: two pairings, keep the cheaper; the axis flip is the same set
# ---------------------------------------------------------------------------
def fig_swap_quotient() -> None:
    xi1, l1 = _dir(22), (0.0, 1.5)
    xi2, l2 = _dir(58), (-0.5, 1.2)
    P1 = (l1[0] * xi1, l1[1] * xi1)
    P2 = (l2[0] * xi2, l2[1] * xi2)
    direct = math.sqrt(np.sum((P1[0] - P2[0]) ** 2) + np.sum((P1[1] - P2[1]) ** 2))
    swapped = math.sqrt(np.sum((P1[0] - P2[1]) ** 2) + np.sum((P1[1] - P2[0]) ** 2))
    pair1 = make_endpoint_pair(np.r_[xi1, 0, 0, 0, 0], *l1)
    pair2 = make_endpoint_pair(np.r_[xi2, 0, 0, 0, 0], *l2)
    E = _E(pair1, pair2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    titles = [
        rf"direct pairing: $\sqrt{{\|z_1^--z_2^-\|^2+\|z_1^+-z_2^+\|^2}} = {direct:.2f}$",
        rf"swapped pairing: $\sqrt{{\|z_1^--z_2^+\|^2+\|z_1^+-z_2^-\|^2}} = {swapped:.2f}$",
        rf"axis flip $(-\xi_1, [-l_1^+, -l_1^-])$: the same segment",
    ]
    for k, ax in enumerate(axes):
        for xi, (lo, hi), color in ((xi1, l1, AMBER), (xi2, l2, BLUE)):
            if k == 2 and color == BLUE:
                continue
            p, q = lo * xi, hi * xi
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=5, solid_capstyle="round", zorder=3)
            ax.scatter([p[0], q[0]], [p[1], q[1]], color=color, s=60, zorder=4, edgecolor="white")
        if k < 2:
            pairs = [(P1[0], P2[0]), (P1[1], P2[1])] if k == 0 else [(P1[0], P2[1]), (P1[1], P2[0])]
            for a, b in pairs:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, linestyle="--", linewidth=1.3, zorder=2)
            ax.text(P1[0][0] + 0.05, P1[0][1] - 0.15, r"$z_1^-$", color=AMBER); ax.text(P1[1][0] + 0.05, P1[1][1] + 0.04, r"$z_1^+$", color=AMBER)
            ax.text(P2[0][0] - 0.28, P2[0][1] - 0.05, r"$z_2^-$", color=BLUE); ax.text(P2[1][0] - 0.05, P2[1][1] + 0.08, r"$z_2^+$", color=BLUE)
        else:
            # the flipped encoding: arrow along -xi, limits negated -> endpoints identical
            ax.annotate("", xy=-xi1 * 0.55 + P1[1], xytext=P1[1], arrowprops=dict(arrowstyle="-|>", color=BRICK, lw=1.6))
            ax.text(*(P1[1] - xi1 * 0.75 + np.array([0.0, 0.12])), r"$-\xi_1$", color=BRICK, fontsize=13)
            ax.text(P1[0][0] + 0.05, P1[0][1] - 0.18, r"$-l_1^+ \cdot(-\xi_1) = z_1^-$", color=BRICK, fontsize=11)
            ax.text(P1[1][0] - 0.95, P1[1][1] + 0.12, r"$-l_1^- \cdot(-\xi_1) = z_1^+$", color=BRICK, fontsize=11)
            ax.text(-0.5, -0.5, "unordered pair $\\{z^-, z^+\\}$ is unchanged, so the\ndirect/swapped minimum absorbs the flip", color=INK2, fontsize=10.5)
        ax.scatter([0], [0], color=INK, s=40, zorder=5)
        ax.set_xlim(-0.6, 1.9); ax.set_ylim(-0.75, 1.55); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k], fontsize=11.5)
        _clean(ax, grid=False)
    fig.suptitle(rf"$E(J_1, J_2) = \min(\mathrm{{direct}}, \mathrm{{swapped}}) = {E:.2f}$", fontsize=14, y=1.02)
    _save(fig, "swap_quotient.png")


# ---------------------------------------------------------------------------
# 4. Revolute 2 pi wrap: the orbit of an interval and the seam of the wrap
# ---------------------------------------------------------------------------
def fig_revolute_wrap() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    lo, hi = 0.0, math.pi / 2
    # number line with the orbit
    ax1.axhline(0, color=INK, linewidth=1)
    ax1.axvspan(-math.pi, math.pi, color=AMBER, alpha=0.12, lw=0)
    ax1.text(0, 0.62, "midpoints wrap into $[-\\pi, \\pi)$", ha="center", color=INK2, fontsize=11)
    for k, color in ((-1, GREY), (0, AMBER), (1, GREY), (2, GREY)):
        a, b = lo + 2 * math.pi * k, hi + 2 * math.pi * k
        y = 0.0
        ax1.plot([a, b], [y, y], color=color, linewidth=7, solid_capstyle="butt", zorder=3)
        ax1.plot([(a + b) / 2], [y], marker="v", color=INK, markersize=7, zorder=4)
        ax1.text((a + b) / 2, -0.22, rf"$[{a/math.pi:.2g}\pi, {b/math.pi:.2g}\pi]$".replace("0\\pi", "0"), ha="center", fontsize=10, color=INK2)
        if k != 0:
            ax1.annotate("", xy=((lo + hi) / 2 + 0.25 * np.sign(k), 0.3), xytext=((a + b) / 2, 0.3),
                         arrowprops=dict(arrowstyle="-|>", color=BRICK, lw=1.2, connectionstyle="arc3,rad=-0.25"))
    wa, wb = wrap_revolute_limits(lo + 2 * math.pi, hi + 2 * math.pi)
    ax1.text(math.pi + 0.3, 0.42, rf"wrap: $k=\lfloor(\mathrm{{mid}}+\pi)/2\pi\rfloor$, both limits $-2\pi k$" + "\n" + rf"e.g. $[2\pi, 2.5\pi] \to [{wa:.2g}, {wb/math.pi:.2g}\pi]$", fontsize=10.5, color=BRICK)
    ax1.set_xlim(-2.5 * math.pi, 5.3 * math.pi); ax1.set_ylim(-0.5, 0.8)
    ax1.set_xticks([k * math.pi for k in range(-2, 6)]); ax1.set_xticklabels([rf"${k}\pi$" if k else "0" for k in range(-2, 6)])
    ax1.set_yticks([])
    ax1.set_xlabel(r"joint coordinate $q$ (rad): every interval on this $2\pi$ orbit is the SAME motion")
    ax1.set_title("One orbit of revolute limits and its canonical representative")
    _clean(ax1, grid=False)

    # E vs shift, raw vs wrapped, using the library
    xi = screw_twist(np.array([0, 0, 1.0]), np.array([0.3, 0.2, 0.0]), "revolute")
    gt = Joint(xi_hat=xi, a=lo, b=hi, joint_type="revolute")
    g = joint_to_endpoint_pair(gt)
    thetas = np.linspace(0, 2 * math.pi, 721)
    raw, wrapped = [], []
    for th in thetas:
        pred = Joint(xi_hat=xi, a=lo + th, b=hi + th, joint_type="revolute")
        p_raw = make_endpoint_pair(xi, pred.a, pred.b)
        p_wr = joint_to_endpoint_pair(pred, wrap_revolute=True)
        raw.append(_E(g, p_raw)); wrapped.append(_E(g, p_wr))
    deg = np.degrees(thetas)
    ax2.plot(deg, raw, color=AMBER, linewidth=2.5, label="raw limits (default)")
    ax2.plot(deg, wrapped, color=GREY, linestyle="--", linewidth=2, label="midpoint wrapped (optional)")
    seam = 135
    ax2.axvline(seam, color=BRICK, linestyle=":", linewidth=1.2)
    ax2.text(seam + 4, max(wrapped) * 0.97, "seam: midpoint\ncrosses $\\pi$", color=BRICK, fontsize=10.5, va="top")
    ax2.set_xlabel(r"shift $\theta$ of both limits (deg), GT $[0, \pi/2]$ vs $[\theta, \pi/2+\theta]$")
    ax2.set_ylabel(r"$E_\alpha$")
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax2.set_title("With the optional wrap a full turn returns E to 0")
    _clean(ax2)
    _save(fig, "revolute_wrap.png")


# ---------------------------------------------------------------------------
# 5. Prismatic origin gauge: the origin never enters a prismatic twist
# ---------------------------------------------------------------------------
def fig_prismatic_gauge() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    a = np.array([1.0, 0.0, 0.0])
    o1, o2 = np.array([0.0, 0.0, 0.0]), np.array([0.6, 0.55, 0.0])
    for ax, jtype in zip(axes, ("prismatic", "revolute")):
        limits = (0.0, 0.8) if jtype == "prismatic" else (0.0, math.pi / 2)
        j1 = Joint(xi_hat=screw_twist(a, o1, jtype), a=limits[0], b=limits[1], joint_type=jtype)
        j2 = Joint(xi_hat=screw_twist(a, o2, jtype), a=limits[0], b=limits[1], joint_type=jtype)
        E = _E(joint_to_endpoint_pair(j1), joint_to_endpoint_pair(j2))
        for o, color, tag in ((o1, AMBER, "1"), (o2, BLUE, "2")):
            ax.annotate("", xy=o[:2] + 0.9 * a[:2], xytext=o[:2], arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
            ax.scatter(*o[:2], color=color, s=60, zorder=4)
            ax.text(o[0] - 0.05, o[1] + 0.1, rf"$\mathbf{{o}}_{tag}$", color=color, fontsize=13)
            xi = (j1 if tag == "1" else j2).xi_hat.data
            fmt = lambda w: "(" + ", ".join(f"{c:g}" for c in np.round(w, 2) + 0.0) + ")"
            ax.text(o[0] + 0.95, o[1] - 0.02, rf"$\xi_{tag} = ({fmt(xi[:3])},\ {fmt(xi[3:])})$", color=color, fontsize=10)
        if jtype == "prismatic":
            ax.text(0.0, -0.45, "same direction, drawn from different origins:\nidentical twists, so the metric sees the same joint", color=INK2, fontsize=11)
        else:
            ax.text(0.0, -0.45, "same direction, different pivot lines:\nthe moment $\\mathbf{o}\\times\\mathbf{a}$ differs, and so does the motion", color=INK2, fontsize=11)
        ax.set_title(rf"{jtype}: $E(J_1, J_2) = {E:.2f}$", fontsize=12.5)
        ax.set_xlim(-0.4, 3.1); ax.set_ylim(-0.7, 1.1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        _clean(ax, grid=False)
    fig.suptitle("Prismatic origin gauge: moving the drawn origin of a slider changes nothing", fontsize=13, y=1.0)
    _save(fig, "prismatic_gauge.png")


# ---------------------------------------------------------------------------
# 6. Split norm vs kinetic-energy norm: the same axis error on links of different size
# ---------------------------------------------------------------------------
def _rod_inertia(L: float, m: float = 1.0) -> LinkInertia:
    """Uniform thin rod of length L along +x, hinged at the origin (one end)."""
    com = np.array([L / 2, 0.0, 0.0])
    # about the com: I_yy = I_zz = m L^2 / 12, tiny I_xx to stay SPD
    I = np.diag([1e-6 * m * L * L, m * L * L / 12, m * L * L / 12])
    return LinkInertia(mass=m, com=com, inertia=I)


def fig_inner_products() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    err_deg = 15.0
    z, o = np.array([0, 0, 1.0]), np.zeros(3)
    tilted = np.array([0.0, -math.sin(math.radians(err_deg)), math.cos(math.radians(err_deg))])
    # drawing: two rods with the same angular error at the hinge
    for L, color, y0 in ((0.35, TEAL, 0.0), (1.0, BLUE, -0.75)):
        ax1.plot([0, L], [y0, y0], color=color, linewidth=9, solid_capstyle="butt")
        ax1.plot([0, L * math.cos(math.radians(err_deg))], [y0, y0 + L * math.sin(math.radians(err_deg))], color=BRICK, linewidth=2, linestyle="--")
        ax1.scatter([0], [y0], color=INK, s=50, zorder=5)
        ax1.text(L + 0.05, y0, f"L = {L:g} m", color=color, va="center")
    ax1.text(0.02, 0.15, rf"axis error ${err_deg:g}^\circ$", color=BRICK)
    ax1.set_xlim(-0.15, 1.6); ax1.set_ylim(-1.05, 0.45); ax1.set_aspect("equal")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("same angular error,\nshort vs long link", fontsize=12)
    _clean(ax1, grid=False)

    Ls = np.linspace(0.1, 2.0, 60)
    e_alpha, e_B = [], []
    for L in Ls:
        gt = Joint(xi_hat=screw_twist(z, o, "revolute"), a=0.0, b=math.pi / 2, joint_type="revolute")
        pr = Joint(xi_hat=screw_twist(tilted, o, "revolute"), a=0.0, b=math.pi / 2, joint_type="revolute")
        pg, pp = joint_to_endpoint_pair(gt), joint_to_endpoint_pair(pr)
        e_alpha.append(_E(pg, pp, SPLIT))
        e_B.append(_E(pg, pp, make_kinetic_energy_norm(_rod_inertia(L))))
    ax2.plot(Ls, e_alpha, color=BLUE, linestyle="--", linewidth=2.2, label=r"$E_\alpha$ (split norm, $\alpha=1$): flat")
    ax2.plot(Ls, e_B, color=BRICK, linewidth=2.5, label=r"$E_B$ (kinetic-energy norm): grows with the link")
    ax2.set_xlabel("link length L (m)"); ax2.set_ylabel("E")
    ax2.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax2.set_title(rf"${err_deg:g}^\circ$ axis error, range $[0, \pi/2]$," + "\nuniform rod hinged at one end", fontsize=12)
    _clean(ax2)
    _save(fig, "inner_products.png")


# ---------------------------------------------------------------------------
# 7. E_B as material motion: displacement field of a door under two hinges
# ---------------------------------------------------------------------------
def _door_inertia(w: float, h: float, m: float = 1.0) -> LinkInertia:
    """Uniform thin rectangular door in the x-y plane, hinge at the origin, spanning [0,w] x [0,h]."""
    com = np.array([w / 2, h / 2, 0.0])
    Ixx, Iyy = m * h * h / 12, m * w * w / 12
    I = np.diag([Ixx, Iyy, Ixx + Iyy])
    return LinkInertia(mass=m, com=com, inertia=I)


def fig_material_motion() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
    w, h, q = 0.8, 1.6, math.radians(60)
    d = 0.18  # pivot offset of the prediction along +x
    gx, gy = np.meshgrid(np.linspace(0.06, w - 0.06, 6), np.linspace(0.06, h - 0.06, 11))
    pts = np.c_[gx.ravel(), gy.ravel()]

    def rot(p, ang, pivot):
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, -s], [s, c]])
        return (p - pivot) @ R.T + pivot

    p_gt = rot(pts, q, np.array([0.0, 0.0]))
    p_pr = rot(pts, q, np.array([d, 0.0]))
    ax1.add_patch(Rectangle((0, 0), w, h, facecolor=HAIR, edgecolor=GREY, lw=1))
    ax1.text(w / 2, h + 0.06, "rest pose", ha="center", color=INK2, fontsize=10)
    for corner_set, color, lab in ((rot(np.array([[0, 0], [w, 0], [w, h], [0, h]]), q, np.array([0.0, 0.0])), AMBER, "GT hinge"),
                                   (rot(np.array([[0, 0], [w, 0], [w, h], [0, h]]), q, np.array([d, 0.0])), BLUE, "predicted hinge")):
        ax1.add_patch(Polygon(corner_set, closed=True, facecolor="none", edgecolor=color, lw=2))
    for a, b in zip(p_gt, p_pr):
        ax1.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=BRICK, lw=0.9, shrinkA=0, shrinkB=0))
    ax1.scatter(p_gt[:, 0], p_gt[:, 1], color=AMBER, s=8, zorder=4)
    ax1.scatter([0], [0], color=AMBER, s=50, zorder=5); ax1.scatter([d], [0], color=BLUE, s=50, zorder=5)
    rms = math.sqrt(np.mean(np.sum((p_gt - p_pr) ** 2, axis=1)))
    ax1.text(0.9, 1.75, f"same opening angle\nq = 60°, hinge offset\nd = {d:g} m\n\nRMS displacement of\nthe material at q\n= {rms:.3f} m", color=INK, fontsize=10.5, va="top")
    ax1.set_xlim(-1.5, 2.1); ax1.set_ylim(-0.2, 1.9); ax1.set_aspect("equal")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(handles=[plt.Line2D([], [], color=AMBER, lw=2, label="GT hinge"), plt.Line2D([], [], color=BLUE, lw=2, label="predicted hinge"),
                        plt.Line2D([], [], color=BRICK, lw=1, label="displacement of each material point")], loc="lower left", frameon=False, fontsize=10)
    ax1.set_title("what $E_B$ measures: where the material ends up")
    _clean(ax1, grid=False)

    # E_B (library, linearized endpoints) vs finite-displacement RMS over the two endpoints
    ds = np.linspace(0, 0.5, 26)
    eB, rms_fin = [], []
    ip = make_kinetic_energy_norm(_door_inertia(w, h))
    z = np.array([0, 0, 1.0])
    for dd in ds:
        gt = Joint(xi_hat=screw_twist(z, np.zeros(3), "revolute"), a=0.0, b=q, joint_type="revolute")
        pr = Joint(xi_hat=screw_twist(z, np.array([dd, 0, 0]), "revolute"), a=0.0, b=q, joint_type="revolute")
        eB.append(_E(joint_to_endpoint_pair(gt), joint_to_endpoint_pair(pr), ip))
        disp_hi = np.mean(np.sum((rot(pts, q, np.array([0.0, 0.0])) - rot(pts, q, np.array([dd, 0.0]))) ** 2, axis=1))
        # E_B^2 sums the two endpoints in quadrature; the lower one (q = 0) is shared, so it contributes 0
        rms_fin.append(math.sqrt(0.0 + disp_hi))
    ax2.plot(ds, eB, color=BRICK, linewidth=2.5, label=r"$E_B$ (endpoint twists, linearized)")
    ax2.plot(ds, rms_fin, color=INK2, linestyle="--", linewidth=1.8, label="RMS of the true finite displacements\n(endpoints summed in quadrature)")
    ax2.set_xlabel("hinge offset d (m)"); ax2.set_ylabel("meters")
    ax2.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax2.set_title(r"$E_B$ reads in meters of material motion")
    _clean(ax2)
    _save(fig, "material_motion.png")


# ---------------------------------------------------------------------------
# 8. Compactification: tanh radial map, the ball, continuous joints as the limit
# ---------------------------------------------------------------------------
def fig_compactification() -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))
    r = np.linspace(0, 12, 300)
    for kappa, color, klab in ((1.0, MAUVE, "1"), (math.pi, AMBER, r"\pi"), (6.0, TEAL, "6")):
        ax1.plot(r, np.tanh(r / kappa), color=color, linewidth=2.2, label=rf"$\kappa = {klab}$")
    ax1.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax1.text(0.3, 1.02, "boundary = continuous joints", color=INK2, fontsize=10)
    ax1.set_xlabel(r"$\|z\|$ (norm of an endpoint)"); ax1.set_ylabel(r"radius after $\phi$: $\tanh(\|z\|/\kappa)$")
    ax1.set_ylim(0, 1.12); ax1.legend(frameon=False, fontsize=10.5, loc="lower right")
    ax1.set_title(r"radial map $\phi(z) = \tanh(\|z\|/\kappa)\,z/\|z\|$", fontsize=12)
    _clean(ax1)

    kappa = math.pi
    ax2.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=INK, lw=1.5))
    xi6 = np.r_[_dir(35), 0, 0, 0, 0]
    xi2 = _dir(35)
    for i, (L, color) in enumerate(((0.6, "#dcd6c8"), (1.5, "#c9c0ab"), (3.0, "#a89c80"), (7.0, "#7a6d52"))):
        u = radial_compactify(-L * xi6, kappa, SPLIT)[:2]; v = radial_compactify(L * xi6, kappa, SPLIT)[:2]
        ax2.plot([u[0], v[0]], [u[1], v[1]], color=color, linewidth=6 - i, solid_capstyle="round", zorder=4 - i)
        ax2.text(1.12, 0.95 - 0.2 * i, rf"$L={L:g}$", color=color, fontsize=10)
        ax2.plot([v[0], 1.08], [v[1], 0.98 - 0.2 * i], color=color, linewidth=0.6)
    cu, cv = continuous_joint_repr(xi6, SPLIT)
    ax2.scatter([cu[0], cv[0]], [cu[1], cv[1]], color=BRICK, s=70, zorder=5, edgecolor="white")
    ax2.text(-1.3, -1.15, r"continuous joint: $\{\pm\xi/\|\xi\|\}$ on the sphere", color=BRICK, fontsize=10)
    # a small finite joint well inside
    p = radial_compactify(np.r_[_dir(-40) * 0.0, 0, 0, 0, 0] + 0, kappa, SPLIT)
    q1, q2 = radial_compactify(np.r_[_dir(-40) * 0.2, 0, 0, 0, 0], kappa, SPLIT)[:2], radial_compactify(np.r_[_dir(-40) * 1.4, 0, 0, 0, 0], kappa, SPLIT)[:2]
    ax2.plot([q1[0], q2[0]], [q1[1], q2[1]], color=BLUE, linewidth=4, solid_capstyle="round")
    ax2.text(q2[0] + 0.05, q2[1] - 0.12, "a finite joint\n(well inside)", color=BLUE, fontsize=9.5)
    ax2.scatter([0], [0], color=INK, s=30)
    ax2.set_xlim(-1.35, 1.75); ax2.set_ylim(-1.3, 1.3); ax2.set_aspect("equal")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(r"unit ball ($\kappa=\pi$): $[-L,L]\,\xi$ grows outward", fontsize=12)
    _clean(ax2, grid=False)

    Ls = np.linspace(0, 14, 200)
    xi = screw_twist(np.array([0, 0, 1.0]), np.zeros(3), "revolute")  # ||xi||_alpha = 1
    cu, cv = continuous_joint_repr(xi.data, SPLIT)
    Ephi = []
    for L in Ls:
        u, v = make_endpoint_pair(xi, -L, L).u.data, make_endpoint_pair(xi, -L, L).v.data
        # phi maps only the finite pair; the continuous pair already sits on the boundary
        pu, pv = radial_compactify(u, kappa, SPLIT), radial_compactify(v, kappa, SPLIT)
        Ephi.append(float(joint_metric(pu, pv, cu, cv, SPLIT)))
    ax3.plot(Ls, Ephi, color=AMBER, linewidth=2.5, label=r"library $E^\phi$")
    ax3.plot(Ls, math.sqrt(2) * (1 - np.tanh(Ls / kappa)), color=INK, linestyle=":", linewidth=1.5, label=r"closed form $\sqrt{2}\,(1-\tanh(L\|\xi\|/\kappa))$")
    ax3.set_xlabel(r"half-range $L$ of the finite prediction (rad)"); ax3.set_ylabel(r"$E^\phi$ vs a continuous GT")
    ax3.legend(frameon=False, fontsize=10.5)
    ax3.set_title("continuous joint = limit of a growing range", fontsize=12)
    _clean(ax3)
    _save(fig, "compactification.png")


# ---------------------------------------------------------------------------
# 9. Tree edit distance: the three operations as a storyboard
# ---------------------------------------------------------------------------
def _draw_tree(ax, nodes: dict, edges: list, colors: dict | None = None, labels: dict | None = None, faded: set | None = None):
    colors = colors or {}
    labels = labels or {}
    faded = faded or set()
    for (u, v) in edges:
        pu, pv = nodes[u], nodes[v]
        c = colors.get((u, v), INK)
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=c, linewidth=4 if c != INK else 2.2, alpha=0.35 if (u, v) in faded else 1, zorder=2)
        if (u, v) in labels:
            mx, my = (pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2
            dx, dy = pv[0] - pu[0], pv[1] - pu[1]
            n = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / n * 0.3, dx / n * 0.3  # perpendicular offset
            if oy < 0:
                ox, oy = -ox, -oy
            ha = "left" if abs(dx) < 1e-9 else "center"  # vertical edge: label hangs to the right
            ax.text(mx + ox, my + oy, labels[(u, v)], color=c if c != INK else INK2, fontsize=10.5, ha=ha, va="center", zorder=4)
    for n, p in nodes.items():
        ax.add_patch(Circle(p, 0.07, facecolor="white", edgecolor=INK, lw=1.5, zorder=3))
        if len(n) == 1:
            ax.text(p[0], p[1], n, ha="center", va="center", fontsize=9, zorder=5)
        elif any(nodes[m][1] < p[1] for m in nodes) and any(nodes[m][1] > p[1] for m in nodes):
            ax.text(p[0] + 0.13, p[1], n, ha="left", va="center", fontsize=10, zorder=5, color=INK)  # mid level: beside
        elif any(nodes[m][1] < p[1] for m in nodes):  # root: name goes above
            ax.text(p[0], p[1] + 0.14, n, ha="center", va="bottom", fontsize=10, zorder=5, color=INK)
        else:
            ax.text(p[0], p[1] - 0.14, n, ha="center", va="top", fontsize=10, zorder=5, color=INK)
    ax.set_xlim(-0.6, 2.7); ax.set_ylim(-0.5, 2.35); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_tree_edit() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    nodes_a = {"base": (1.0, 2.0), "door": (0.2, 1.0), "drawer": (1.8, 1.0), "handle": (0.2, 0.0)}
    edges_a = [("base", "door"), ("base", "drawer"), ("door", "handle")]
    _draw_tree(axes[0], nodes_a, edges_a, colors={("door", "handle"): BRICK}, labels={("base", "door"): r"$J_{\rm door}$", ("base", "drawer"): r"$J_{\rm drawer}$", ("door", "handle"): r"$J_\epsilon$ (nearly welded)"})
    axes[0].set_title("prediction $T_1$", fontsize=12)
    nodes_b = {"base": (1.0, 2.0), "door+handle": (0.2, 1.0), "drawer": (1.8, 1.0)}
    edges_b = [("base", "door+handle"), ("base", "drawer")]
    _draw_tree(axes[1], nodes_b, edges_b, labels={("base", "door+handle"): r"$J_{\rm door}$", ("base", "drawer"): r"$J_{\rm drawer}$"})
    axes[1].set_title(r"contract $J_\epsilon$: cost $E(J_\epsilon, J_0) = \epsilon\|\xi\|$", fontsize=12, color=BRICK)
    _draw_tree(axes[2], nodes_b, edges_b, colors={("base", "drawer"): AMBER}, labels={("base", "door+handle"): r"$J_{\rm door}$", ("base", "drawer"): r"$J_{\rm drawer} \to J'_{\rm drawer}$"})
    axes[2].set_title(r"substitute: cost $E(J_{\rm drawer}, J'_{\rm drawer})$", fontsize=12, color=AMBER)
    nodes_c = {"base": (1.0, 2.0), "door+handle": (0.2, 1.0), "drawer": (1.8, 1.0), "shelf": (1.8, 0.0)}
    edges_c = [("base", "door+handle"), ("base", "drawer"), ("drawer", "shelf")]
    _draw_tree(axes[3], nodes_c, edges_c, colors={("drawer", "shelf"): TEAL}, labels={("base", "door+handle"): r"$J_{\rm door}$", ("base", "drawer"): r"$J'_{\rm drawer}$", ("drawer", "shelf"): r"$J_{\rm shelf}$"})
    axes[3].set_title(r"expand: cost $E(J_{\rm shelf}, J_0)$  $\Rightarrow$ ground truth $T_2$", fontsize=12, color=TEAL)
    fig.suptitle(r"$E^{\rm tree}(T_1, T_2)$ = cheapest edit sequence; every cost is an $E$ between joints, $J_0 = \{0,0\}$ is the welded joint", fontsize=13, y=1.03)
    _save(fig, "tree_edit.png")


# ---------------------------------------------------------------------------
# 10. Relaxation vs exact: same joint multiset, different topology
# ---------------------------------------------------------------------------
def fig_tree_relaxation() -> None:
    z = np.array([0, 0, 1.0])
    small = make_endpoint_pair(screw_twist(z, np.zeros(3), "revolute"), 0.0, 0.3)
    big = make_endpoint_pair(screw_twist(z, np.zeros(3), "revolute"), 0.0, 1.5)
    sp, bp = (small.u.data, small.v.data), (big.u.data, big.v.data)
    path = [("A", "B", sp), ("B", "C", bp), ("C", "D", sp)]
    star = [("A", "B", sp), ("A", "C", sp), ("A", "D", bp)]
    relax = assignment_tree_distance([e[2] for e in path], [e[2] for e in star], SPLIT)
    exact, certified = exact_tree_distance(path, star, SPLIT)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nodes_p = {"A": (0.0, 1.0), "B": (0.8, 1.0), "C": (1.6, 1.0), "D": (2.4, 1.0)}
    _draw_tree(axes[0], nodes_p, [("A", "B"), ("B", "C"), ("C", "D")],
               colors={("B", "C"): BRICK}, labels={("A", "B"): "small", ("B", "C"): "BIG", ("C", "D"): "small"})
    axes[0].set_title("path: small, BIG, small", fontsize=12)
    nodes_s = {"A": (1.2, 1.2), "B": (0.2, 0.3), "C": (1.2, 0.2), "D": (2.2, 0.3)}
    _draw_tree(axes[1], nodes_s, [("A", "B"), ("A", "C"), ("A", "D")],
               colors={("A", "D"): BRICK}, labels={("A", "B"): "small", ("A", "C"): "small", ("A", "D"): "BIG"})
    axes[1].set_title("star: small, small, BIG", fontsize=12)
    fig.text(0.5, -0.06,
             f"same multiset of joints, so the assignment relaxation pairs them one-to-one for free: relaxation = {relax:.3f}\n"
             f"no tree isomorphism realizes that pairing, so the exact edit distance must move motion around: exact = {exact:.3f} "
             f"(certified by the relaxation: {certified})",
             ha="center", fontsize=11.5, color=INK)
    _save(fig, "tree_relaxation.png")


if __name__ == "__main__":
    fig_twist_embeddings()
    fig_endpoint_pair()
    fig_swap_quotient()
    fig_revolute_wrap()
    fig_prismatic_gauge()
    fig_inner_products()
    fig_material_motion()
    fig_compactification()
    fig_tree_edit()
    fig_tree_relaxation()
