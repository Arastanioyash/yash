"""Generate knob-by-knob tuning graphs from face pipeline telemetry CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_float(raw: str) -> float:
    if raw is None:
        return float("nan")
    stripped = str(raw).strip()
    if not stripped:
        return float("nan")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def parse_int(raw: str) -> int | None:
    value = parse_float(raw)
    if not math.isfinite(value):
        return None
    return int(value)


def parse_bool(raw: str) -> bool:
    stripped = str(raw).strip()
    return stripped == "1" or stripped.lower() == "true"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def pick_config(rows: list[dict[str, str]], key: str, default: float = float("nan")) -> float:
    for row in rows:
        value = parse_float(row.get(key, ""))
        if math.isfinite(value):
            return value
    return default


def numeric_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = np.asarray([parse_float(row.get(key, "")) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_knob_hist(
    values: np.ndarray,
    knob_value: float,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if values.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for this knob.", ha="center", va="center")
        save_plot(fig, out_path)
        return

    ax.hist(values, bins=40, alpha=0.75)
    if math.isfinite(knob_value):
        ax.axvline(knob_value, color="crimson", linestyle="--", label="current knob")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    save_plot(fig, out_path)


def plot_single_knob_line(
    values: np.ndarray,
    knob_value: float,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if values.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for this knob.", ha="center", va="center")
        save_plot(fig, out_path)
        return

    ax.plot(values, linewidth=1.2)
    if math.isfinite(knob_value):
        ax.axhline(knob_value, color="crimson", linestyle="--", label="current knob")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel(ylabel)
    save_plot(fig, out_path)


def annotate_knob_value(ax: plt.Axes, label: str, value: float) -> None:
    if math.isfinite(value):
        ax.text(
            0.02,
            0.96,
            f"{label}={value:.4g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
        )


def scalar_from_rows(rows: list[dict[str, str]], key: str) -> np.ndarray:
    value = pick_config(rows, key)
    if not math.isfinite(value):
        return np.asarray([], dtype=np.float64)
    return np.asarray([value], dtype=np.float64)


def plot_per_knob_graphs(
    rows: list[dict[str, str]],
    face_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    flush_rows: list[dict[str, str]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    detect_rows = [row for row in frame_rows if parse_bool(row.get("detect_ran", ""))]

    min_face_sizes = np.asarray(
        [
            min(parse_float(row.get("face_w", "")), parse_float(row.get("face_h", "")))
            for row in face_rows
        ],
        dtype=np.float64,
    )
    min_face_sizes = min_face_sizes[np.isfinite(min_face_sizes)]

    # 1) MATCH_THRESHOLD
    path = out_dir / "match_threshold.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "best_score"),
        knob_value=pick_config(rows, "config_match_threshold"),
        title="MATCH_THRESHOLD",
        xlabel="best_score",
        out_path=path,
    )
    generated.append(path)

    # 2) MARGIN
    path = out_dir / "margin.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "margin_gap"),
        knob_value=pick_config(rows, "config_margin"),
        title="MARGIN",
        xlabel="best_score - second_score",
        out_path=path,
    )
    generated.append(path)

    # 3) COMMIT_EVERY_N
    path = out_dir / "commit_every_n.png"
    plot_single_knob_hist(
        values=numeric_array(flush_rows, "pending_ops_before_flush"),
        knob_value=pick_config(rows, "config_commit_every_n"),
        title="COMMIT_EVERY_N",
        xlabel="pending_ops_before_flush",
        out_path=path,
    )
    generated.append(path)

    # 4) COMMIT_EVERY_SEC
    path = out_dir / "commit_every_sec.png"
    plot_single_knob_hist(
        values=numeric_array(flush_rows, "elapsed_since_commit_sec"),
        knob_value=pick_config(rows, "config_commit_every_sec"),
        title="COMMIT_EVERY_SEC",
        xlabel="elapsed_since_commit_sec",
        out_path=path,
    )
    generated.append(path)

    # 5) DETECT_EVERY_N
    path = out_dir / "detect_every_n.png"
    detect_signal = np.asarray(
        [1.0 if parse_bool(row.get("detect_ran", "")) else 0.0 for row in frame_rows],
        dtype=np.float64,
    )
    detect_every_n = pick_config(rows, "config_detect_every_n")
    plot_single_knob_line(
        values=detect_signal,
        knob_value=1.0,
        title="DETECT_EVERY_N (detect_ran over frames)",
        ylabel="detect_ran",
        out_path=path,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    if detect_signal.size:
        ax.plot(detect_signal, linewidth=1.2)
        ax.axhline(1.0, color="crimson", linestyle="--", label="detect runs")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        annotate_knob_value(ax, "detect_every_n", detect_every_n)
        ax.set_title("DETECT_EVERY_N (detect_ran over frames)")
        ax.set_xlabel("frame index")
        ax.set_ylabel("detect_ran")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for this knob.", ha="center", va="center")
    save_plot(fig, path)
    generated.append(path)

    # 6) DETECT_FRAME_SCALE
    path = out_dir / "detect_frame_scale.png"
    detect_frame_scale = pick_config(rows, "config_detect_frame_scale")
    plot_single_knob_hist(
        values=numeric_array(detect_rows, "detect_ms"),
        knob_value=float("nan"),
        title=f"DETECT_FRAME_SCALE={detect_frame_scale:.3f} vs detect_ms",
        xlabel="detect_ms",
        out_path=path,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    detect_ms_values = numeric_array(detect_rows, "detect_ms")
    if detect_ms_values.size:
        ax.hist(detect_ms_values, bins=40, alpha=0.75)
        ax.set_title("DETECT_FRAME_SCALE vs detect_ms")
        ax.set_xlabel("detect_ms")
        ax.set_ylabel("count")
        annotate_knob_value(ax, "detect_frame_scale", detect_frame_scale)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for this knob.", ha="center", va="center")
    save_plot(fig, path)
    generated.append(path)

    # 7) DET_SIZE
    path = out_dir / "det_size.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    det_w = pick_config(rows, "config_det_size_w")
    det_h = pick_config(rows, "config_det_size_h")
    if math.isfinite(det_w) and math.isfinite(det_h):
        ax.bar(["det_width", "det_height"], [det_w, det_h])
        ax.set_title("DET_SIZE")
        ax.set_ylabel("pixels")
        ax.text(
            0.02,
            0.96,
            f"det_size=({int(det_w)},{int(det_h)})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
        )
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No DET_SIZE config in telemetry.", ha="center", va="center")
    save_plot(fig, path)
    generated.append(path)

    # 8) DETECTION_SCORE_THRESHOLD
    path = out_dir / "detection_score_threshold.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "det_score"),
        knob_value=pick_config(rows, "config_detection_score_threshold"),
        title="DETECTION_SCORE_THRESHOLD",
        xlabel="det_score",
        out_path=path,
    )
    generated.append(path)

    # 9) MIN_FACE_SIZE_PX
    path = out_dir / "min_face_size_px.png"
    plot_single_knob_hist(
        values=min_face_sizes,
        knob_value=pick_config(rows, "config_min_face_size_px"),
        title="MIN_FACE_SIZE_PX",
        xlabel="min(face_w, face_h)",
        out_path=path,
    )
    generated.append(path)

    # 10) MAX_FACE_ASPECT_RATIO
    path = out_dir / "max_face_aspect_ratio.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "aspect_ratio"),
        knob_value=pick_config(rows, "config_max_face_aspect_ratio"),
        title="MAX_FACE_ASPECT_RATIO",
        xlabel="aspect_ratio",
        out_path=path,
    )
    generated.append(path)

    # 11) MIN_FACE_AREA_RATIO
    path = out_dir / "min_face_area_ratio.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "area_ratio"),
        knob_value=pick_config(rows, "config_min_face_area_ratio"),
        title="MIN_FACE_AREA_RATIO",
        xlabel="area_ratio",
        out_path=path,
    )
    generated.append(path)

    # 12) CENTROID_ALPHA
    path = out_dir / "centroid_alpha.png"
    centroid_alpha = pick_config(rows, "config_centroid_alpha")
    plot_single_knob_hist(
        values=numeric_array(face_rows, "embedding_update_l2"),
        knob_value=float("nan"),
        title=f"CENTROID_ALPHA={centroid_alpha:.3f} vs embedding_update_l2",
        xlabel="embedding_update_l2",
        out_path=path,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    step_values = numeric_array(face_rows, "embedding_update_l2")
    if step_values.size:
        ax.hist(step_values, bins=40, alpha=0.75)
        ax.set_title("CENTROID_ALPHA vs embedding_update_l2")
        ax.set_xlabel("embedding_update_l2")
        ax.set_ylabel("count")
        annotate_knob_value(ax, "centroid_alpha", centroid_alpha)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data for this knob.", ha="center", va="center")
    save_plot(fig, path)
    generated.append(path)

    # 13) NEW_PERSON_SAVE_DELAY_SEC
    path = out_dir / "new_person_save_delay_sec.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "pending_remaining_sec"),
        knob_value=pick_config(rows, "config_new_person_save_delay_sec"),
        title="NEW_PERSON_SAVE_DELAY_SEC",
        xlabel="pending_remaining_sec",
        out_path=path,
    )
    generated.append(path)

    # 14) PENDING_NEW_SIM_THRESHOLD
    path = out_dir / "pending_new_sim_threshold.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "pending_similarity"),
        knob_value=pick_config(rows, "config_pending_new_sim_threshold"),
        title="PENDING_NEW_SIM_THRESHOLD",
        xlabel="pending_similarity",
        out_path=path,
    )
    generated.append(path)

    # 15) PENDING_NEW_STALE_SEC
    path = out_dir / "pending_new_stale_sec.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "pending_remaining_sec"),
        knob_value=pick_config(rows, "config_pending_new_stale_sec"),
        title="PENDING_NEW_STALE_SEC",
        xlabel="pending_remaining_sec",
        out_path=path,
    )
    generated.append(path)

    # 16) MIN_BLUR_LAPLACIAN_VAR
    path = out_dir / "min_blur_laplacian_var.png"
    plot_single_knob_hist(
        values=numeric_array(face_rows, "blur_score"),
        knob_value=pick_config(rows, "config_min_blur_laplacian_var"),
        title="MIN_BLUR_LAPLACIAN_VAR",
        xlabel="blur_score",
        out_path=path,
    )
    generated.append(path)

    return generated


def plot_detection_knobs(
    face_rows: list[dict[str, str]],
    config_rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    if not face_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No face rows in telemetry.", ha="center", va="center")
        save_plot(fig, out_path)
        return

    valid_mask = np.asarray([parse_bool(row.get("face_valid", "")) for row in face_rows], dtype=bool)
    det_scores = numeric_array(face_rows, "det_score")
    min_face_sizes = np.asarray(
        [
            min(parse_float(row.get("face_w", "")), parse_float(row.get("face_h", "")))
            for row in face_rows
        ],
        dtype=np.float64,
    )
    min_face_sizes = min_face_sizes[np.isfinite(min_face_sizes)]
    aspect_ratios = numeric_array(face_rows, "aspect_ratio")
    area_ratios = numeric_array(face_rows, "area_ratio")
    blur_scores = numeric_array(face_rows, "blur_score")

    det_threshold = pick_config(config_rows, "config_detection_score_threshold")
    min_face_threshold = pick_config(config_rows, "config_min_face_size_px")
    max_aspect_threshold = pick_config(config_rows, "config_max_face_aspect_ratio")
    min_area_threshold = pick_config(config_rows, "config_min_face_area_ratio")
    min_blur_threshold = pick_config(config_rows, "config_min_blur_laplacian_var")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    axes[0].hist(det_scores, bins=40, alpha=0.7, label="all faces")
    if math.isfinite(det_threshold):
        axes[0].axvline(det_threshold, color="crimson", linestyle="--", label="threshold")
    axes[0].set_title("Detection Score")
    axes[0].set_xlabel("det_score")
    axes[0].legend()

    axes[1].hist(min_face_sizes, bins=40, alpha=0.7)
    if math.isfinite(min_face_threshold):
        axes[1].axvline(min_face_threshold, color="crimson", linestyle="--")
    axes[1].set_title("Min Face Size (px)")
    axes[1].set_xlabel("min(face_w, face_h)")

    axes[2].hist(aspect_ratios, bins=40, alpha=0.7)
    if math.isfinite(max_aspect_threshold):
        axes[2].axvline(max_aspect_threshold, color="crimson", linestyle="--")
    axes[2].set_title("Aspect Ratio")
    axes[2].set_xlabel("max(w/h, h/w)")

    axes[3].hist(area_ratios, bins=40, alpha=0.7)
    if math.isfinite(min_area_threshold):
        axes[3].axvline(min_area_threshold, color="crimson", linestyle="--")
    axes[3].set_title("Face Area Ratio")
    axes[3].set_xlabel("face_area / frame_area")

    axes[4].hist(blur_scores, bins=40, alpha=0.7)
    if math.isfinite(min_blur_threshold) and min_blur_threshold > 0:
        axes[4].axvline(min_blur_threshold, color="crimson", linestyle="--")
    axes[4].set_title("Blur Score (Laplacian Var)")
    axes[4].set_xlabel("blur_score")

    reject_counts = Counter(row.get("reject_reason", "unknown") for row in face_rows if not parse_bool(row.get("face_valid", "")))
    if reject_counts:
        labels = list(reject_counts.keys())
        values = [reject_counts[label] for label in labels]
        axes[5].bar(labels, values)
        axes[5].tick_params(axis="x", rotation=30)
        axes[5].set_title("Reject Reasons")
    else:
        axes[5].text(0.5, 0.5, "No rejections", ha="center", va="center")
        axes[5].set_title("Reject Reasons")

    kept_rate = float(valid_mask.mean()) if valid_mask.size else 0.0
    fig.suptitle(f"Detection Knobs (face keep rate: {kept_rate:.1%})")
    save_plot(fig, out_path)


def plot_matching_knobs(
    face_rows: list[dict[str, str]],
    config_rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    if not face_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No face rows in telemetry.", ha="center", va="center")
        save_plot(fig, out_path)
        return

    best_scores = numeric_array(face_rows, "best_score")
    margin_gaps = numeric_array(face_rows, "margin_gap")
    pending_similarity = numeric_array(face_rows, "pending_similarity")
    pending_remaining = numeric_array(face_rows, "pending_remaining_sec")
    update_steps = numeric_array(face_rows, "embedding_update_l2")

    match_threshold = pick_config(config_rows, "config_match_threshold")
    margin_threshold = pick_config(config_rows, "config_margin")
    pending_sim_threshold = pick_config(config_rows, "config_pending_new_sim_threshold")
    delay_threshold = pick_config(config_rows, "config_new_person_save_delay_sec")

    new_person_count = sum(parse_bool(row.get("is_new_person", "")) for row in face_rows)
    matched_count = sum(bool(str(row.get("matched_person_id", "")).strip()) for row in face_rows)
    unknown_count = len(face_rows) - matched_count

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    axes[0].hist(best_scores, bins=40, alpha=0.7)
    if math.isfinite(match_threshold):
        axes[0].axvline(match_threshold, color="crimson", linestyle="--")
    axes[0].set_title("Best Match Score")
    axes[0].set_xlabel("best_score")

    axes[1].hist(margin_gaps, bins=40, alpha=0.7)
    if math.isfinite(margin_threshold):
        axes[1].axvline(margin_threshold, color="crimson", linestyle="--")
    axes[1].set_title("Match Margin Gap")
    axes[1].set_xlabel("best_score - second_score")

    axes[2].hist(pending_similarity, bins=40, alpha=0.7)
    if math.isfinite(pending_sim_threshold):
        axes[2].axvline(pending_sim_threshold, color="crimson", linestyle="--")
    axes[2].set_title("Pending Similarity")
    axes[2].set_xlabel("pending_similarity")

    axes[3].hist(pending_remaining, bins=40, alpha=0.7)
    if math.isfinite(delay_threshold):
        axes[3].axvline(delay_threshold, color="crimson", linestyle="--")
    axes[3].set_title("Pending Remaining Time")
    axes[3].set_xlabel("seconds")

    axes[4].hist(update_steps, bins=40, alpha=0.7)
    axes[4].set_title("Centroid Update L2 Step")
    axes[4].set_xlabel("embedding_update_l2")

    axes[5].bar(
        ["matched", "unknown", "new_person_saved"],
        [matched_count, unknown_count, new_person_count],
    )
    axes[5].set_title("Recognition Outcomes")

    fig.suptitle("Matching + New-Person Gate Knobs")
    save_plot(fig, out_path)


def plot_runtime_knobs(
    frame_rows: list[dict[str, str]],
    flush_rows: list[dict[str, str]],
    config_rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    if not frame_rows and not flush_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No frame/flush rows in telemetry.", ha="center", va="center")
        save_plot(fig, out_path)
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    frame_indices = np.asarray(
        [parse_int(row.get("frame_idx", "")) for row in frame_rows if parse_int(row.get("frame_idx", "")) is not None],
        dtype=np.int64,
    )
    detect_ran = np.asarray(
        [1 if parse_bool(row.get("detect_ran", "")) else 0 for row in frame_rows],
        dtype=np.int64,
    )
    detect_ms = numeric_array([row for row in frame_rows if parse_bool(row.get("detect_ran", ""))], "detect_ms")
    faces_seen = numeric_array([row for row in frame_rows if parse_bool(row.get("detect_ran", ""))], "faces_seen")
    faces_kept = numeric_array([row for row in frame_rows if parse_bool(row.get("detect_ran", ""))], "faces_kept")

    if frame_indices.size and detect_ran.size == frame_indices.size:
        axes[0].plot(frame_indices, detect_ran, linewidth=1.2)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_title("Detect Cadence by Frame")
    axes[0].set_xlabel("frame_idx")
    axes[0].set_ylabel("detect_ran")

    axes[1].plot(detect_ms, linewidth=1.2)
    axes[1].set_title("Detection Latency")
    axes[1].set_xlabel("detect step")
    axes[1].set_ylabel("ms")

    if faces_seen.size and faces_kept.size:
        axes[2].plot(faces_seen, label="faces_seen", linewidth=1.2)
        axes[2].plot(faces_kept, label="faces_kept", linewidth=1.2)
        axes[2].legend()
    axes[2].set_title("Faces Seen vs Kept")
    axes[2].set_xlabel("detect step")

    pending_ops = numeric_array(flush_rows, "pending_ops_before_flush")
    elapsed = numeric_array(flush_rows, "elapsed_since_commit_sec")
    if pending_ops.size and elapsed.size:
        n = min(pending_ops.size, elapsed.size)
        axes[3].scatter(pending_ops[:n], elapsed[:n], s=16, alpha=0.75)
    axes[3].set_title("Flush Trigger Space")
    axes[3].set_xlabel("pending_ops_before_flush")
    axes[3].set_ylabel("elapsed_since_commit_sec")

    flush_reasons = Counter(row.get("flush_reason", "unknown") for row in flush_rows)
    if flush_reasons:
        labels = list(flush_reasons.keys())
        values = [flush_reasons[label] for label in labels]
        axes[4].bar(labels, values)
        axes[4].tick_params(axis="x", rotation=25)
    axes[4].set_title("Flush Reasons")

    detect_every_n = pick_config(config_rows, "config_detect_every_n")
    detect_frame_scale = pick_config(config_rows, "config_detect_frame_scale")
    det_w = pick_config(config_rows, "config_det_size_w")
    det_h = pick_config(config_rows, "config_det_size_h")
    commit_every_n = pick_config(config_rows, "config_commit_every_n")
    commit_every_sec = pick_config(config_rows, "config_commit_every_sec")
    centroid_alpha = pick_config(config_rows, "config_centroid_alpha")
    text = (
        "Runtime knobs\n"
        f"detect_every_n={detect_every_n:.0f}\n"
        f"detect_frame_scale={detect_frame_scale:.3f}\n"
        f"det_size=({det_w:.0f},{det_h:.0f})\n"
        f"commit_every_n={commit_every_n:.0f}\n"
        f"commit_every_sec={commit_every_sec:.2f}\n"
        f"centroid_alpha={centroid_alpha:.3f}"
    )
    axes[5].axis("off")
    axes[5].text(0.03, 0.95, text, va="top", family="monospace")

    fig.suptitle("Runtime + Commit Knobs")
    save_plot(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("artifacts/tuning_events.csv"),
        help="CSV produced by hshub tuning logger.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/tuning_plots"),
        help="Directory where PNG graphs are written.",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"Telemetry file not found: {args.log}")

    rows = load_rows(args.log)
    if not rows:
        raise RuntimeError(f"Telemetry file has no rows: {args.log}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    face_rows = [row for row in rows if row.get("event_type") == "face"]
    frame_rows = [row for row in rows if row.get("event_type") == "frame"]
    flush_rows = [row for row in rows if row.get("event_type") == "flush"]

    detection_path = args.out_dir / "detection_knobs.png"
    matching_path = args.out_dir / "matching_knobs.png"
    runtime_path = args.out_dir / "runtime_knobs.png"

    plot_detection_knobs(face_rows=face_rows, config_rows=rows, out_path=detection_path)
    plot_matching_knobs(face_rows=face_rows, config_rows=rows, out_path=matching_path)
    plot_runtime_knobs(
        frame_rows=frame_rows,
        flush_rows=flush_rows,
        config_rows=rows,
        out_path=runtime_path,
    )
    per_knob_paths = plot_per_knob_graphs(
        rows=rows,
        face_rows=face_rows,
        frame_rows=frame_rows,
        flush_rows=flush_rows,
        out_dir=args.out_dir / "per_knob",
    )

    print("Generated tuning graphs:")
    print(f"- {detection_path}")
    print(f"- {matching_path}")
    print(f"- {runtime_path}")
    print("- per-knob:")
    for per_path in per_knob_paths:
        print(f"  - {per_path}")


if __name__ == "__main__":
    main()
