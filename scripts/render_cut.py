# scripts/render_cut.py
#
# The cut a stroke site run left, from the bed's heights site_run.py saves
# after every stroke (<video>_cut.npz, StrokeSite.save): the ground in 3D
# through Newton's GL viewer, coloured and lit as the run's video, true to
# scale; and sections against the targets: along the strokes' line, averaged
# across the cut's width, once with depth stretched and once true to scale
# with the rover drawn in it, digging; and across each step, its lanes side
# by side together, averaged along it. Writes <name>_3d.png,
# <name>_sections.png and <name>_portfolio.png (the true-to-scale section
# and the steps alone, ground level dotted) beside it.
#
#   ~/IsaacLab/.venv/bin/python scripts/render_cut.py deep60_video_cut.npz

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


def _robot():
    """The robot model's dimensions (luna_hifi_tasks/excavator/excavator.py),
    loaded by its file: the package around it would start Isaac Lab."""
    path = Path(__file__).resolve().parent.parent / "luna_hifi_tasks" / "excavator" / "excavator.py"
    spec = importlib.util.spec_from_file_location("excavator_dims", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["excavator_dims"] = m
    spec.loader.exec_module(m)
    return m


EX = _robot()

MARGIN = 0.8                    # m of ground drawn round the strokes
# The soil's colours, its depth shading and the light, as the run's video has them
# (excavate_rft_env's SOIL_RGB, PILE_RGB, DEPTH_DARK, SHADE_BLUR, PILE_RISE; site_run's light()).
SOIL, PILE, DARK = (0.62, 0.55, 0.45), (0.80, 0.73, 0.58), (0.31, 0.275, 0.225)
SHADE_BLUR, PILE_RISE = 0.2, 0.025
SUN_AZIMUTH, SUN_ELEVATION, EXPOSURE, SHADOWS_M = -20.0, 35.0, 1.0, 20.0
SIZE, FOCAL = (2400, 1350), 20.0          # the 3D view's pixels, and its lens, mm
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LINE, TARGET = "#2a78d6", "#898781"
# The rover in the true-to-scale section: its wheels, its frame and arms, its drums.
WHEEL, BODY, DRUM = "#3b3a37", "#6f6e69", "#a8a6a0"
ARM_DIG_START = -0.20           # rad, the arm's angle as a pass starts (excavate_env_cfg.arm_start_angle)


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


def rover_pose(g: np.ndarray, x: np.ndarray, dig_x: float) -> dict:
    """The rover side on, to scale (excavator.py), digging with its front
    drum's axis over dig_x at the arm's reach a pass starts from: its wheels
    resting on the ground g along x, its front drum lowered onto the ground
    there as deep as its arm goes, its rear drum at the pass's start pose.
    The chassis origin, its pitch, the front arm's angle and the top of it all."""
    def ground(at):
        return np.interp(at, x, g)

    def rest(at: float, r: float) -> float:
        """The height a circle of radius r centred over `at` rests at on the ground."""
        dx = np.linspace(-0.98 * r, 0.98 * r, 41)
        return float(np.max(ground(at + dx) + np.sqrt(r * r - dx * dx)))

    cx = dig_x - (EX.PIVOT_X + EX.ARM_LEN * math.cos(ARM_DIG_START))
    zr, zf = (rest(cx + s * EX.AXLE_X, EX.WHEEL_RADIUS) for s in (-1.0, 1.0))
    pose = dict(cx=cx, cz=0.5 * (zr + zf), pitch=math.atan2(zf - zr, 2.0 * EX.AXLE_X), front=EX.ARM_RANGE[0])
    for angle in np.linspace(EX.ARM_RANGE[0], EX.ARM_RANGE[1], 300):
        px, pz = _world(pose, *_drum_centre(1.0, angle))
        if pz >= rest(float(px), EX.SHROUD_OUT_R) - 1e-3:
            pose["front"] = angle
    rear = float(_world(pose, *_drum_centre(-1.0, ARM_DIG_START))[1]) + EX.SHROUD_OUT_R
    pose["top"] = max(rear, pose["cz"] + EX.WHEEL_RADIUS)
    return pose


def _drum_centre(sign: float, angle: float) -> tuple[float, float]:
    """A drum's axis in the chassis frame for an arm angle, front (sign 1) or rear (-1); positive pitches the boom down."""
    return sign * (EX.PIVOT_X + EX.ARM_LEN * math.cos(angle)), EX.MAST_TOP_Z - EX.ARM_LEN * math.sin(angle)


def _world(pose: dict, px, pz):
    """Chassis-frame points (x forward, z up from the axle line) in the section."""
    c, s = math.cos(pose["pitch"]), math.sin(pose["pitch"])
    px, pz = np.asarray(px, float), np.asarray(pz, float)
    return pose["cx"] + c * px - s * pz, pose["cz"] + s * px + c * pz


def draw_rover(ax, pose: dict) -> None:
    """The rover in its pose (rover_pose): its frame and masts, arms and drums, wheels; labelled."""
    def box(x0, x1, z0, z1):
        xs, zs = _world(pose, [x0, x1, x1, x0], [z0, z0, z1, z1])
        ax.add_patch(Polygon(np.stack([xs, zs], -1), closed=True, fc=BODY, ec="none", zorder=4))

    (fx, _, fz), (hx, _, hz) = EX.FRAME_BOXES[0]
    box(fx - hx, fx + hx, fz - hz, fz + hz)
    for (mx, _, mz), (hx, _, hz) in EX.FRAME_BOXES[1:]:
        box(mx - hx, mx + hx, mz - hz, mz + hz)
    for sign, angle in ((1.0, pose["front"]), (-1.0, ARM_DIG_START)):
        dx, dz = _drum_centre(sign, angle)
        xs, zs = _world(pose, [sign * EX.PIVOT_X, dx], [EX.MAST_TOP_Z, dz])
        ax.plot(xs, zs, color=BODY, lw=3, solid_capstyle="round", zorder=5)
        ax.add_patch(Circle((xs[1], zs[1]), EX.SHROUD_OUT_R, fc=DRUM, ec=BODY, lw=1, zorder=6))
    for sign in (-1.0, 1.0):
        wx, wz = _world(pose, sign * EX.AXLE_X, 0.0)
        ax.add_patch(Circle((float(wx), float(wz)), EX.WHEEL_RADIUS, fc=WHEEL, ec="none", zorder=6))
        ax.add_patch(Circle((float(wx), float(wz)), 0.35 * EX.WHEEL_RADIUS, fc=BODY, ec="none", zorder=7))
    ax.text(pose["cx"], pose["top"] + 0.1, "the rover, to scale", ha="center", va="bottom", fontsize=9.5, color=INK2)


def section_data(c: dict) -> dict:
    """What the sections draw: the ground along x averaged across the cut's
    width, the steps, the target floor, and the rover standing on the
    deepest stroke near its far end, digging there; the true-to-scale
    section's height range."""
    z, x, y, rects = c["z"], c["x"], c["y"], c["rects"]
    inside = (y >= rects[:, 2].min()) & (y <= rects[:, 3].max())
    g = z[inside].mean(axis=0)
    deepest = int(np.argmin(c["targets"]))
    pose = rover_pose(g, x, float(rects[deepest][1]) - 0.3)
    tx, tz = staircase(c)
    z_lo = min(float(g.min()), float(c["targets"].min())) - 0.15
    return dict(inside=inside, g=g, steps=steps(c), tx=tx, tz=tz, pose=pose,
                z_lo=z_lo, z_hi=max(0.5, pose["top"] + 0.35))


def draw_true_scale(ax, c: dict, d: dict) -> None:
    """Along the trench, true to scale: the soil, the ground averaged across
    the cut's width, the target floor, ground level dotted bold, the rover."""
    x = c["x"]
    ax.fill_between(x, d["z_lo"], d["g"], color=SOIL, lw=0, zorder=1)
    ax.plot(x, d["g"], color=LINE, lw=1.5, zorder=2)
    ax.plot(d["tx"], d["tz"], color=TARGET, lw=1, ls="--", zorder=3)
    ax.axhline(0.0, color=INK, lw=2.4, ls=":", zorder=3.5)
    ax.text(x[0] + 0.1, 0.03, "ground level", ha="left", va="bottom", fontsize=9.5, color=INK, zorder=3.5)
    draw_rover(ax, d["pose"])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(d["z_lo"], d["z_hi"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Along the trench, true to scale", loc="left", fontsize=12, fontweight="semibold", color=INK, pad=10)
    ax.set_xlabel("along x (m)")
    ax.set_ylabel("height (m)")


def draw_steps(fig, c: dict, d: dict, left: float, right: float, bottom: float, top: float) -> None:
    """Across each step: the ground from side to side averaged along the step,
    against its target, ground level dotted bold."""
    z, x, y = c["z"], c["x"], c["y"]
    grid = fig.add_gridspec(1, len(d["steps"]), wspace=0.12, left=left, right=right, bottom=bottom, top=top)
    lo = 100 * min(z[d["inside"]].min(), c["targets"].min()) - 5
    for col, (x0, x1, y0, y1, t, name) in enumerate(d["steps"]):
        ax = fig.add_subplot(grid[0, col])
        on = (x >= x0) & (x <= x1)
        ax.plot(y, 100 * z[:, on].mean(axis=1), color=LINE, lw=2, zorder=3)
        ax.plot([y[0], y0, y0, y1, y1, y[-1]], [0, 0, 100 * t, 100 * t, 0, 0], color=TARGET, lw=1.5, zorder=2)
        ax.axhline(0.0, color=INK, lw=2.0, ls=":", zorder=4)
        ax.set_title(f"{name}\nx = {x0:.1f} to {x1:.1f} m", fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(lo, 12)
        ax.set_xlabel("across y (m)")
        if col:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("height (cm)")


def style(fig) -> None:
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.xaxis.label.set_color(INK2)
        ax.yaxis.label.set_color(INK2)


def _layout(rows: list[float], width_in: float):
    """A figure of rows (inches, top down), and row k's bottom and height as figure fractions."""
    height = sum(rows)
    fig = plt.figure(figsize=(width_in, height), dpi=200, facecolor=SURFACE)
    return fig, height, lambda k: (1.0 - sum(rows[:k + 1]) / height, rows[k] / height)


LEFT, RIGHT, WIDTH_IN = 0.06, 0.98, 14.0


def _scale_h(c: dict, d: dict) -> float:
    """The true-to-scale section's height in inches, as wide as the figure's others."""
    return (RIGHT - LEFT) * WIDTH_IN * (d["z_hi"] - d["z_lo"]) / (c["x"][-1] - c["x"][0])


def render_sections(c: dict, out: str) -> None:
    """Along the strokes' line: the ground averaged across the cut's width,
    depth stretched, and the same true to scale with the rover drawn in it;
    across each step: the ground from side to side, averaged along the step.
    All against the targets."""
    x, d = c["x"], section_data(c)
    fig, height, band = _layout([1.7, 2.4, 0.95, _scale_h(c, d), 1.1, 2.4, 0.7], WIDTH_IN)
    fig.text(LEFT, 1.0 - 0.45 / height, f"Cross sections after {c['moves']} strokes", fontsize=16,
             fontweight="semibold", color=INK)
    fig.text(LEFT, 1.0 - 0.8 / height, "the ground as the run left it (blue), averaged across the cut's width or along "
             "each step, against the target floor (grey)", fontsize=11, color=INK2)

    b, h = band(1)
    along = fig.add_axes([LEFT, b, RIGHT - LEFT, h])
    along.plot(x, 100 * d["g"], color=LINE, lw=2, label="the ground, averaged across the cut's width")
    along.plot(d["tx"], 100 * np.asarray(d["tz"]), color=TARGET, lw=1.5, label="target floor")
    for x0, x1, _, _, _, name in d["steps"]:
        along.text(0.5 * (x0 + x1), 4, name, ha="center", fontsize=10, color=INK2)
    along.set_title("Along the trench, depth stretched", loc="left", fontsize=12, fontweight="semibold", color=INK,
                    pad=22)
    along.set_xlim(x[0], x[-1])
    along.set_xlabel("along x (m)")
    along.set_ylabel("height (cm)")
    along.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2, ncol=2)

    b, h = band(3)
    draw_true_scale(fig.add_axes([LEFT, b, RIGHT - LEFT, h]), c, d)
    b, h = band(5)
    draw_steps(fig, c, d, LEFT, RIGHT, b, b + h)
    style(fig)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def render_portfolio(c: dict, out: str) -> None:
    """The trench true to scale with the rover in it, and across each step;
    ground level dotted bold in both."""
    d = section_data(c)
    fig, height, band = _layout([1.4, _scale_h(c, d), 1.1, 2.4, 0.7], WIDTH_IN)
    deep = -float(c["targets"].min())
    fig.text(LEFT, 1.0 - 0.45 / height, f"A {100 * deep:.0f} cm trench down a staircase", fontsize=16,
             fontweight="semibold", color=INK)
    fig.text(LEFT, 1.0 - 0.8 / height, "the ground the rover left (blue), averaged across the trench or along each step, "
             "against the target floor (grey); ground level dotted", fontsize=11, color=INK2)
    b, h = band(1)
    draw_true_scale(fig.add_axes([LEFT, b, RIGHT - LEFT, h]), c, d)
    b, h = band(3)
    draw_steps(fig, c, d, LEFT, RIGHT, b, b + h)
    style(fig)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Draw a stroke site's cut from site_run.py's saved heights.")
    p.add_argument("cut", help="the .npz site_run.py saved, <video>_cut.npz")
    path = p.parse_args().cut
    c, stem = load(path), str(Path(path).with_suffix(""))
    render_sections(c, f"{stem}_sections.png")
    print(f"{stem}_sections.png", flush=True)
    render_portfolio(c, f"{stem}_portfolio.png")
    print(f"{stem}_portfolio.png", flush=True)
    render_3d(c, f"{stem}_3d.png")
    print(f"{stem}_3d.png")


if __name__ == "__main__":
    main()
