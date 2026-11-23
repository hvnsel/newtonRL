# scripts/render_cut.py
#
# The cut a stroke site run left, from the bed's heights site_run.py saves
# after every stroke (<video>_cut.npz, StrokeSite.save): the ground in 3D
# through Newton's GL viewer, coloured and lit as the run's video, true to
# scale; and cross sections against the targets, along the strokes' line and
# across each step's middle, its lanes side by side together. Writes
# <name>_3d.png and <name>_sections.png beside it.
#
#   ~/IsaacLab/.venv/bin/python scripts/render_cut.py deep60_video_cut.npz

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MARGIN = 0.8                    # m of ground drawn round the strokes
# The soil's colours, its depth shading and the light, as the run's video has them
# (excavate_rft_env's SOIL_RGB, PILE_RGB, DEPTH_DARK, SHADE_BLUR, PILE_RISE; site_run's light()).
SOIL, PILE, DARK = (0.62, 0.55, 0.45), (0.80, 0.73, 0.58), (0.31, 0.275, 0.225)
SHADE_BLUR, PILE_RISE = 0.2, 0.025
SUN_AZIMUTH, SUN_ELEVATION, EXPOSURE, SHADOWS_M = -20.0, 35.0, 1.0, 20.0
SIZE, FOCAL = (2400, 1350), 20.0          # the 3D view's pixels, and its lens, mm
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LINE, TARGET = "#2a78d6", "#898781"


def load(path: str) -> dict:
    """The saved heights (m against the ground as it was) cropped to the strokes
    and MARGIN round them, the cells' centres, and the strokes."""
    d = np.load(path)
    rects, cell = d["rects"], float(d["cell"])
    x0, x1 = rects[:, 0].min() - MARGIN, rects[:, 1].max() + MARGIN
    y0, y1 = rects[:, 2].min() - MARGIN, rects[:, 3].max() + MARGIN
    ny, nx = d["z"].shape
    x = d["lower"][0] + (np.arange(nx) + 0.5) * cell
    y = d["lower"][1] + (np.arange(ny) + 0.5) * cell
    i, j = (x >= x0) & (x <= x1), (y >= y0) & (y <= y1)
    return dict(z=d["z"][np.ix_(j, i)].astype(float), x=x[i], y=y[j], cell=cell, rects=rects, targets=d["targets"],
                names=[str(n) for n in d["names"]], moves=int(d["moves"]))


def colours(c: dict) -> np.ndarray:
    """(ny, nx, 3) as the video colours the soil: darker with depth down to the
    deepest target, over the strokes and 0.3 m round them, the depth averaged
    over SHADE_BLUR; soil PILE_RISE above the ground a pile's colour."""
    z, x, y, rects = c["z"], c["x"], c["y"], c["rects"]
    k = max(1, round(SHADE_BLUR / c["cell"])) | 1
    pad = np.pad(-z, k // 2, mode="edge")
    soft = np.lib.stride_tricks.sliding_window_view(pad, (k, k)).mean(axis=(-2, -1))
    X, Y = np.meshgrid(x, y)
    near = np.zeros_like(z, dtype=bool)
    for x0, x1, y0, y1 in rects:
        near |= (X >= x0 - 0.3) & (X <= x1 + 0.3) & (Y >= y0 - 0.3) & (Y <= y1 + 0.3)
    shade = np.clip(soft / -c["targets"].min(), 0.0, 1.0) * near
    rgb = np.asarray(SOIL) + shade[..., None] * (np.asarray(DARK) - np.asarray(SOIL))
    return np.where(((z >= PILE_RISE) & (shade < 0.025))[..., None], np.asarray(PILE), rgb)


def render_3d(c: dict, out: str) -> None:
    """The ground as one textured mesh in Newton's GL viewer, from above the
    strokes' near end and to their side, true to scale; titled."""
    headless = not os.environ.get("DISPLAY")
    import pyglet
    pyglet.options["headless"] = headless           # EGL without a display, as Isaac Lab's Newton visualizer
    import newton
    import newton.viewer
    import warp as wp
    from pyglet.math import Vec3

    z, x, y, rects = c["z"], c["x"], c["y"], c["rects"]
    ny, nx = z.shape
    v = newton.viewer.ViewerGL(width=SIZE[0], height=SIZE[1], headless=headless)
    v.set_model(newton.ModelBuilder(up_axis=newton.Axis.Z).finalize(device="cpu"))
    r = v.renderer
    r.draw_shadows, r.draw_sky = True, False
    r.sky_upper = r.sky_lower = tuple(int(SURFACE[i:i + 2], 16) / 255 for i in (1, 3, 5))
    a, e = math.radians(SUN_AZIMUTH), math.radians(SUN_ELEVATION)
    r.exposure, r.shadow_extents = EXPOSURE, SHADOWS_M
    r._sun_direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    bx0, bx1, by0, by1 = rects[:, 0].min(), rects[:, 1].max(), rects[:, 2].min(), rects[:, 3].max()
    length, deep = bx1 - bx0, -float(c["targets"].min())
    v.camera.pos = Vec3(bx0 - 0.08 * length, by0 - 0.38 * length, 0.3 * length)
    v.camera.look_at(Vec3(bx0 + 0.6 * length, 0.5 * (by0 + by1), -0.8 * deep))
    v.camera.fov = math.degrees(2.0 * math.atan(15.2908 / (2.0 * FOCAL)))

    # The mesh on the viewer's device, its model's (the CPU): it reads the arrays
    # there, and one on a GPU would be read by a CPU kernel and crash.
    dev = v.device
    X, Y = np.meshgrid(x, y)
    points = wp.array(np.stack([X, Y, z], -1).reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=dev)
    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    q = (j * nx + i).reshape(-1)
    # Two triangles per square of four cell centres, anticlockwise seen from above.
    tris = wp.array(np.stack([q, q + 1, q + nx + 1, q, q + nx + 1, q + nx], -1).reshape(-1).astype(np.int32),
                    dtype=wp.int32, device=dev)
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    # Texel (j, i) is cell (j, i): the viewer flips an image's rows on upload.
    uvs = wp.array(np.stack([(i + 0.5) / nx, 1.0 - (j + 0.5) / ny], -1).reshape(-1, 2).astype(np.float32),
                   dtype=wp.vec2, device=dev)
    texture = (255.0 * colours(c)).round().astype(np.uint8)
    for _ in range(2):          # log_mesh leaves the texture off: it is turned on once the mesh is made
        v.begin_frame(0.0)
        mesh = v.objects.get("cut")
        if mesh is not None:
            mesh.material = (*mesh.material[:3], 1.0)
        v.log_mesh("cut", points, tris, uvs=uvs, texture=texture, color=SOIL if mesh is None else (1.0, 1.0, 1.0),
                   roughness=0.95, metallic=0.0)
        v.end_frame()
    frame = v.get_frame().numpy()[..., :3]

    fig = plt.figure(figsize=(SIZE[0] / 200, SIZE[1] / 200 + 0.8), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0.0, 0.0, 1.0, SIZE[1] / (SIZE[1] + 160)])
    ax.imshow(frame)
    ax.set_axis_off()
    fig.text(0.03, 0.955, f"The cut after {c['moves']} strokes", fontsize=16, fontweight="semibold", color=INK, va="top")
    fig.text(0.03, 0.905, f"{length:.1f} m long, {by1 - by0:.1f} m wide, true to scale; deepest point "
             f"{100 * -z.min():.0f} cm", fontsize=11, color=INK2, va="top")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def steps(c: dict) -> list[tuple]:
    """The stretches of the strokes' line in order along x, the strokes side by
    side on each taken together: (x0, x1, y0, y1, target, name), their widths
    joined and the lane left off the name."""
    out: dict = {}
    for (x0, x1, y0, y1), t, name in zip(c["rects"], c["targets"], c["names"]):
        key = (round(float(x0), 3), round(float(x1), 3))
        a, b = out[key][2:4] if key in out else (y0, y1)
        out[key] = (x0, x1, min(a, y0), max(b, y1), t, name.split(",")[0].split(" of ")[0])
    return [out[key] for key in sorted(out)]


def staircase(c: dict) -> tuple[list[float], list[float]]:
    """The target floor along x, the steps' targets end to end, up to the ground at either end."""
    st = steps(c)
    xs, zs = [c["x"][0], st[0][0]], [0.0, 0.0]
    for x0, x1, _, _, t, _ in st:
        xs += [x0, x1]
        zs += [t, t]
    return xs + [st[-1][1], c["x"][-1]], zs + [0.0, 0.0]


def render_sections(c: dict, out: str) -> None:
    """Along the strokes' line: the ground at its middle and the range across
    the cut's width; across each step's middle: the ground from side to side.
    Both against the targets."""
    z, x, y, rects = c["z"], c["x"], c["y"], c["rects"]
    mid_y = 0.5 * (rects[:, 2].min() + rects[:, 3].max())
    inside = (y >= rects[:, 2].min()) & (y <= rects[:, 3].max())
    st = steps(c)
    fig = plt.figure(figsize=(14, 9), dpi=200, facecolor=SURFACE)
    grid = fig.add_gridspec(2, len(st), height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.12,
                            left=0.06, right=0.98, top=0.8, bottom=0.08)
    fig.text(0.06, 0.95, f"Cross sections after {c['moves']} strokes", fontsize=16, fontweight="semibold", color=INK)
    fig.text(0.06, 0.915, "the ground as the run left it (blue) against the target floor (grey); depth not to scale "
             "with length", fontsize=11, color=INK2)

    along = fig.add_subplot(grid[0, :])
    j = np.argmin(np.abs(y - mid_y))
    band = z[inside]
    along.fill_between(x, 100 * band.min(axis=0), 100 * band.max(axis=0), color=LINE, alpha=0.12, lw=0,
                       label="range across the cut's width")
    along.plot(x, 100 * z[j], color=LINE, lw=2, label="middle of the cut")
    tx, tz = staircase(c)
    along.plot(tx, 100 * np.asarray(tz), color=TARGET, lw=1.5, label="target floor")
    for x0, x1, _, _, _, name in st:
        along.text(0.5 * (x0 + x1), 4, name, ha="center", fontsize=10, color=INK2)
    along.set_title("Along the trench", loc="left", fontsize=12, fontweight="semibold", color=INK, pad=22)
    along.set_xlabel("along x (m)")
    along.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2, ncol=3)

    lo = 100 * min(z[inside].min(), c["targets"].min()) - 5
    for col, (x0, x1, y0, y1, t, name) in enumerate(st):
        ax = fig.add_subplot(grid[1, col])
        i = np.argmin(np.abs(x - 0.5 * (x0 + x1)))
        ax.plot(y, 100 * z[:, i], color=LINE, lw=2)
        ax.plot([y[0], y0, y0, y1, y1, y[-1]], [0, 0, 100 * t, 100 * t, 0, 0], color=TARGET, lw=1.5)
        ax.set_title(f"{name}, at x = {x[i]:.1f} m", fontsize=10.5, color=INK, loc="left")
        ax.set_ylim(lo, 12)
        ax.set_xlabel("across y (m)")
        if col:
            ax.tick_params(labelleft=False)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.xaxis.label.set_color(INK2)
    fig.axes[0].set_ylabel("height (cm)", color=INK2)
    fig.axes[1].set_ylabel("height (cm)", color=INK2)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Draw a stroke site's cut from site_run.py's saved heights.")
    p.add_argument("cut", help="the .npz site_run.py saved, <video>_cut.npz")
    path = p.parse_args().cut
    c, stem = load(path), str(Path(path).with_suffix(""))
    render_sections(c, f"{stem}_sections.png")
    print(f"{stem}_sections.png", flush=True)
    render_3d(c, f"{stem}_3d.png")
    print(f"{stem}_3d.png")


if __name__ == "__main__":
    main()
