#!/usr/bin/env python3
"""Generate the ArticulateArena icon (static/images/articulatearena-icon.svg).

The mark is a capital A drawn as an articulated linkage: the two legs are two
links from the part palette, the crossbar a third, and the vertices carry the
paper-colored joint dots of the paper's kinematic diagrams. Colors are the
SPARK part palette + the site's warm ink/paper tones, so the icon matches the
project pages. All geometry is computed here; edit the constants and rerun.

Usage:
    python scripts/make_icon.py
"""

from pathlib import Path

# Site palette
INK = "#26231D"      # warm ink, the rounded-square ground
PAPER = "#FCFBF8"    # site paper tone, joint dots
BLUE = "#5B8DB8"     # part palette 1 (left leg)
TEAL = "#6FB5AE"     # part palette 3 (right leg)
AMBER = "#E5B455"    # part palette 2 / site accent (crossbar + pivot)

SIZE = 512
CORNER = 104          # background corner radius (same as the old icon)

APEX = (256.0, 116.0)          # top pivot of the A
FOOT_L = (148.0, 404.0)        # left leg foot
FOOT_R = (364.0, 404.0)        # right leg foot
BAR_T = 0.67                   # crossbar position along each leg (0 = apex)
LEG_W = 58.0                   # leg stroke width
BAR_W = 50.0                   # crossbar stroke width
R_PIVOT = 27.0                 # apex joint dot
R_JOINT = 17.0                 # crossbar joint dots
R_PIVOT_CORE = 12.0            # amber core of the apex pivot


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def line(p, q, color, width):
    return (f'  <line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" '
            f'stroke="{color}" stroke-width="{width:.0f}" stroke-linecap="round"/>')


def circle(c, r, fill):
    return f'  <circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r:.0f}" fill="{fill}"/>'


def main():
    j_l = lerp(APEX, FOOT_L, BAR_T)   # crossbar joints sit ON the legs
    j_r = lerp(APEX, FOOT_R, BAR_T)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}" role="img" aria-labelledby="title desc">',
        '  <title id="title">ArticulateArena icon</title>',
        '  <desc id="desc">A capital A drawn as an articulated linkage: two palette-colored '
        'links joined at an amber pivot, with a crossbar link and joint dots.</desc>',
        f'  <rect x="24" y="24" width="{SIZE - 48}" height="{SIZE - 48}" rx="{CORNER}" fill="{INK}"/>',
        # links (legs under the crossbar, crossbar under the joint dots)
        line(APEX, FOOT_L, BLUE, LEG_W),
        line(APEX, FOOT_R, TEAL, LEG_W),
        line(j_l, j_r, AMBER, BAR_W),
        # joint dots: paper rings, amber pivot core at the apex
        circle(j_l, R_JOINT, PAPER),
        circle(j_r, R_JOINT, PAPER),
        circle(APEX, R_PIVOT, PAPER),
        circle(APEX, R_PIVOT_CORE, AMBER),
        '</svg>',
    ]

    out = Path(__file__).resolve().parent.parent / "static/images/articulatearena-icon.svg"
    out.write_text("\n".join(parts) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
