"""Turn a heightmap dump from play_dig.py into a video of the dig.

    python3 scripts/render_dig.py dig_bed.npz
    python3 scripts/render_dig.py dig_bed.npz --out dig.gif --fps 12

Writes an animated GIF, which needs only matplotlib and plays anywhere, and
optionally the individual PNGs with --png.

Why this rather than a rendered viewport: on the Newton backend env.render()
returns None, the GL viewer runs headless under EGL with no readable buffer,
and there is no display on the workstation anyway. But the soil surface is
already a (ny, nx) heightmap recomputed every step for the reward, so the
thing worth seeing can be drawn from data rather than captured from pixels.
It also shows what a camera would not: height CHANGE against the opening
surface, so removal and spoil piles separate by colour.

Three panels:

  left    the bed from above, coloured by height change. Blue is soil gone,
          red is soil piled up. The work area is outlined, and the machine's
          position and heading are marked.
  top     payload and cumulative removal against time, with the playhead.
  bottom  the surface profile along the machine's own row, opening surface
          dashed, so a hole under the plane is visible as a notch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an (h, w, 3) uint8 array as a PNG, with zlib and nothing else.

    Here so that a box without matplotlib still gets pictures. PNG's own
    format is a zlib stream of rows each prefixed by a filter byte, wrapped in
    four chunks whose lengths and CRCs are big-endian.
    """
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _colour(delta: np.ndarray, span: float) -> np.ndarray:
    """Height change to RGB: blue where soil left, red where it piled up."""
    u = np.clip(delta / max(span, 1e-9), -1.0, 1.0)
    neg, pos = np.clip(-u, 0, 1), np.clip(u, 0, 1)
    # white at no change, saturating either way, so zero reads as background.
    r = 1.0 - 0.85 * neg
    g = 1.0 - 0.85 * (neg + pos)
    b = 1.0 - 0.85 * pos
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _fallback(d, out_dir: Path, scale: int, stride: int) -> int:
    """Top-down frames with numpy only. One picture per saved step."""
    bed = d["bed"][::stride]
    pose = d["pose"][::stride]
    work = d["work_mask"]
    voxel = float(d["voxel"])
    lower = d["lower"]
    T, ny, nx = bed.shape
    delta = bed - bed[0]
    span = float(np.percentile(np.abs(delta), 99.5)) or voxel

    ys, xs = np.nonzero(work) if work.any() else (np.array([0]), np.array([0]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(T):
        img = _colour(delta[k], span)
        # Work-area outline, one cell thick, before upscaling.
        if work.any():
            for yy in (ys.min(), ys.max()):
                img[yy, xs.min():xs.max() + 1] = (0, 0, 0)
            for xx in (xs.min(), xs.max()):
                img[ys.min():ys.max() + 1, xx] = (0, 0, 0)
        # The machine, as a filled square at its grid cell.
        cx = int(np.clip(round((pose[k, 0] - lower[0]) / voxel), 1, nx - 2))
        cy = int(np.clip(round((pose[k, 1] - lower[1]) / voxel), 1, ny - 2))
        img[cy - 1:cy + 2, cx - 1:cx + 2] = (20, 120, 20)
        # origin="lower" by hand: PNG rows run top-down.
        img = img[::-1]
        if scale > 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        _write_png(out_dir / f"{k:04d}.png", img)
    print(f"wrote {T} PNGs to {out_dir} ({nx * scale} x {ny * scale}), "
          f"colour span +-{span * 100:.1f} cm")
    print(f"make a video with:  ffmpeg -framerate 12 -i {out_dir}/%04d.png "
          f"-pix_fmt yuv420p {out_dir}.mp4")
    return 0


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", help="the dump written by play_dig.py --dump")
    p.add_argument("--out", default=None, help="output GIF; default is <npz>.gif")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--png", default=None, help="also write every frame as a PNG here")
    p.add_argument("--stride", type=int, default=1, help="use every Nth saved frame")
    p.add_argument("--scale", type=int, default=4,
                   help="pixel size per voxel in the numpy-only fallback")
    p.add_argument("--no_mpl", action="store_true",
                   help="skip matplotlib and write the plain top-down PNGs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    d = np.load(args.npz)

    if args.no_mpl:
        return _fallback(d, Path(args.npz).with_suffix(""), args.scale, args.stride)
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        print("no matplotlib here; writing the plain top-down frames instead")
        return _fallback(d, Path(args.npz).with_suffix(""), args.scale, args.stride)

    import matplotlib
    matplotlib.use("Agg")                    # no display, and none needed
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    bed = d["bed"][:: args.stride]           # (T, ny, nx)
    pose = d["pose"][:: args.stride]         # (T, 7) x y z yaw t removal payload
    work = d["work_mask"]
    voxel = float(d["voxel"])
    lower = d["lower"]
    cap = float(d["drum_capacity_kg"])

    T, ny, nx = bed.shape
    # Cell centres in env-local metres. soil_heightmap indexes [y][x] from
    # bed_grid_lower, so extent follows that and nothing is transposed.
    x0, y0 = float(lower[0]), float(lower[1])
    extent = (x0, x0 + nx * voxel, y0, y0 + ny * voxel)

    opening = bed[0]
    delta = bed - opening                    # negative where soil left
    # Symmetric scale, ignoring the extremes so one flung cell does not wash
    # the whole map out.
    span = float(np.percentile(np.abs(delta), 99.5)) or voxel
    t = pose[:, 4]
    removal, payload = pose[:, 5], pose[:, 6]

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.38, wspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_mass = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, 1])

    im = ax_map.imshow(delta[0], origin="lower", extent=extent, aspect="equal",
                       cmap="RdBu_r", vmin=-span, vmax=span)
    fig.colorbar(im, ax=ax_map, label="height change (m)   blue = soil removed",
                 fraction=0.035, pad=0.02)
    # The work area, as the env masked it: its own bounding box in cell space.
    if work.any():
        ys, xs = np.nonzero(work)
        ax_map.add_patch(Rectangle(
            (x0 + xs.min() * voxel, y0 + ys.min() * voxel),
            (xs.max() - xs.min() + 1) * voxel, (ys.max() - ys.min() + 1) * voxel,
            fill=False, ec="k", lw=1.6, ls="--", label="work area"))
    machine, = ax_map.plot([], [], "o", ms=11, mfc="none", mec="k", mew=2.2)
    heading, = ax_map.plot([], [], "-", c="k", lw=2.2)
    track, = ax_map.plot([], [], "-", c="0.35", lw=1.0, alpha=0.8)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.legend(loc="upper right", fontsize=8)

    ax_mass.plot(t, removal, label="removed from work area", c="tab:blue")
    ax_mass.plot(t, payload, label="in the drums", c="tab:orange")
    ax_mass.axhline(cap, c="0.5", ls=":", lw=1.2, label=f"drum pair cap {cap:.0f} kg")
    playhead = ax_mass.axvline(t[0], c="k", lw=1.0)
    ax_mass.set_xlabel("t (s)")
    ax_mass.set_ylabel("kg")
    ax_mass.legend(fontsize=8, loc="upper left")

    prof_now, = ax_prof.plot([], [], c="tab:brown", lw=1.6, label="surface now")
    xs_m = x0 + (np.arange(nx) + 0.5) * voxel
    prof_open, = ax_prof.plot([], [], c="0.6", lw=1.0, ls="--", label="opening surface")
    ax_prof.set_xlabel("x (m) along the machine's row")
    ax_prof.set_ylabel("z (m)")
    ax_prof.legend(fontsize=8, loc="lower left")
    lo = float(np.nanmin(bed)), float(np.nanmax(bed))
    ax_prof.set_ylim(lo[0] - 0.02, lo[1] + 0.02)
    ax_prof.set_xlim(extent[0], extent[1])

    def row_of(py: float) -> int:
        return int(np.clip(round((py - y0) / voxel - 0.5), 0, ny - 1))

    def update(k: int):
        im.set_data(delta[k])
        px, py, _, yaw = pose[k, 0], pose[k, 1], pose[k, 2], pose[k, 3]
        machine.set_data([px], [py])
        heading.set_data([px, px + 0.9 * np.cos(yaw)], [py, py + 0.9 * np.sin(yaw)])
        track.set_data(pose[: k + 1, 0], pose[: k + 1, 1])
        playhead.set_xdata([t[k], t[k]])
        r = row_of(py)
        prof_now.set_data(xs_m, bed[k, r])
        prof_open.set_data(xs_m, opening[r])
        ax_map.set_title(f"t = {t[k]:5.2f} s     removed {removal[k]:6.1f} kg     "
                         f"drums {payload[k]:5.1f} kg")
        return im, machine, heading, track, playhead, prof_now, prof_open

    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".gif")
    anim = FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    print(f"wrote {out} ({T} frames at {args.fps} fps)")

    if args.png:
        png_dir = Path(args.png)
        png_dir.mkdir(parents=True, exist_ok=True)
        for k in range(T):
            update(k)
            fig.savefig(png_dir / f"{k:04d}.png", dpi=90)
        print(f"wrote {T} PNGs to {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
