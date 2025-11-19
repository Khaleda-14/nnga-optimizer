# electrode_optimizer/plotting.py
from matplotlib import patches
import numpy as np

def _safe_label(ax, label, handle):
    """
    Add label to handle only if that label isn't already present on the axes.
    Prevents duplicate legend entries when repeatedly redrawing.
    """
    _, labels = ax.get_legend_handles_labels()
    if label not in labels:
        handle.set_label(label)
    else:
        handle.set_label("_nolegend_")  # matplotlib convention to skip legend entry

def draw_electrodes(ax, canvas, area_um, pitch_um, optimized=None):
    """
    Draw two square electrodes (centered) separated by 'pitch' along x-axis.

    - area_um is interpreted as the side length (µm) of each square.
    - optimized can be None or an iterable (opt_side, opt_pitch) or dict with keys 'side'/'pitch'.
      The function is robust to very small side values by enforcing a minimum visible size
      (relative to the plotting pad) so the optimized geometry never looks like a vertical line.
    """
    ax.clear()

    # Interpret inputs
    try:
        side = float(area_um)
    except Exception:
        side = 0.0
    try:
        pitch = float(pitch_um)
    except Exception:
        pitch = 0.0

    # Basic centers for input electrodes (using input side and pitch)
    y_center = 0.0
    left_center_x = - (pitch / 2.0 + side / 2.0)
    right_center_x = + (pitch / 2.0 + side / 2.0)

    # Safety: ensure we will draw something visible even if side is zero/near-zero
    max_dim = max(side, pitch) if max(side, pitch) > 0 else 1.0
    min_vis = max(1e-3, max_dim * 0.02)  # 2% of the largest dimension (or a tiny floor)

    if side <= 0:
        side_to_draw = min_vis
    else:
        side_to_draw = side

    # Input electrode rectangles (blue)
    rects = [
        (left_center_x - side_to_draw / 2.0, y_center - side_to_draw / 2.0, side_to_draw, side_to_draw),
        (right_center_x - side_to_draw / 2.0, y_center - side_to_draw / 2.0, side_to_draw, side_to_draw)
    ]
    for (x, y, w, h) in rects:
        r = patches.Rectangle((x, y), w, h, fill=False, edgecolor='tab:blue', linewidth=2, zorder=1)
        _safe_label(ax, 'Input', r)
        ax.add_patch(r)

    # Draw optimized geometry (if provided) as green dashed squares
    if optimized is not None:
        # Accept several formats for optimized: [side,pitch], (side,pitch), {'side':..,'pitch':..}
        if isinstance(optimized, dict):
            opt_side = float(optimized.get('side', optimized.get('Size', 0.0)))
            opt_pitch = float(optimized.get('pitch', optimized.get('Pitch', 0.0)))
        else:
            try:
                opt_side = float(optimized[0])
                opt_pitch = float(optimized[1])
            except Exception:
                # Fallback: treat optimized as a single value for side (pitch unknown) -> use input pitch
                try:
                    opt_side = float(optimized)
                    opt_pitch = pitch
                except Exception:
                    opt_side, opt_pitch = 0.0, pitch

        # If optimized side is zero or extremely small, make it visible relative to pad
        if opt_side <= 0:
            opt_side_to_draw = min_vis
        else:
            opt_side_to_draw = opt_side

        # Compute centers using optimized pitch and side
        lc_x = - (opt_pitch / 2.0 + opt_side_to_draw / 2.0)
        rc_x = + (opt_pitch / 2.0 + opt_side_to_draw / 2.0)
        rects_opt = [
            (lc_x - opt_side_to_draw / 2.0, y_center - opt_side_to_draw / 2.0, opt_side_to_draw, opt_side_to_draw),
            (rc_x - opt_side_to_draw / 2.0, y_center - opt_side_to_draw / 2.0, opt_side_to_draw, opt_side_to_draw)
        ]
        for (x, y, w, h) in rects_opt:
            r = patches.Rectangle((x, y), w, h, fill=False, edgecolor='tab:green',
                                   linewidth=2, linestyle='--', zorder=2)
            _safe_label(ax, 'Optimized', r)
            ax.add_patch(r)

    # Legend handling: create/update legend if labels present
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right')

    # Set nice axis limits, leave some padding proportional to sizes
    max_dim = max(side_to_draw, pitch, opt_side_to_draw if optimized is not None else 0.0, 1.0)
    pad = max_dim * 2.0 + 20.0
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad / 2.0, pad / 2.0)
    ax.set_aspect('equal', 'box')
    ax.set_title("Electrode Geometry (µm)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    canvas.draw_idle()


def draw_threshold_marker(ax, canvas, area, pitch, pred, label="Input"):
    ax.scatter([0], [pred], label=f"{label}: {pred:.6f} mA")
    ax.set_title("Threshold Predictions")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Threshold (mA)")
    ax.legend()
    canvas.draw_idle()


def draw_threshold_history(ax, canvas, best_history, mean_history, model=None, scaler=None, init=None):
    ax.clear()
    gens = list(range(1, len(best_history) + 1))
    ax.plot(gens, best_history, label='Best Fitness (mA)', marker='o')
    ax.plot(gens, mean_history, label='Mean Fitness (mA)', marker='x')
    ax.set_xlabel("Generation")
    ax.set_ylabel("Threshold (mA)")
    ax.set_title("NNGA Optimizer Progress")
    ax.grid(True)
    ax.legend()

    # If provided, try to draw a horizontal line for the initial prediction.
    # The model/scaler may expect 2 or 3 features; try both gracefully.
    if init is not None and model is not None and scaler is not None:
        init_area, init_pitch, _ = init if len(init) >= 2 else (init[0], 0.0, None)
        init_pred = None
        try:
            # Try 3-feature input (common: [displacement, area, pitch])
            inp = np.array([[100.0, float(init_area), float(init_pitch)]], dtype=float)
            init_pred = float(model.predict(scaler.transform(inp), verbose=0)[0][0])
        except Exception:
            try:
                # Try 2-feature input (area, pitch)
                inp2 = np.array([[float(init_area), float(init_pitch)]], dtype=float)
                init_pred = float(model.predict(scaler.transform(inp2), verbose=0)[0][0])
            except Exception:
                init_pred = None
        if init_pred is not None:
            ax.axhline(init_pred, color='tab:gray', linestyle=':', label=f'Initial Pred {init_pred:.6f}')
            ax.legend()

    canvas.draw_idle()
