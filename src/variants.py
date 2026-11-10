"""The Moonshine variants an admin can choose: model size x number format, each with an estimate of what it takes, and
whether this machine can run it. CPU only: the installed onnxruntime has no GPU provider (the GPU build cannot be
installed beside it), so no GPU variant is offered.

The disk sizes are the real sizes of the files in `UsefulSensors/moonshine` under `onnx/merged/<size>/<format>/` (checked
on the Hub). `quantized_4bit` is not offered: the library loads only `float` and `quantized`.

The estimate is arithmetic, not a measurement: the file size times 1.5 (the runtime holds the weights and its working
buffers) plus 300 MB for onnxruntime. It is shown as an estimate on the admin page.

shortcut: free memory is read when the page is opened; two services starting at once can both see the same free memory."""

from __future__ import annotations

from typing import Any

# (model, number format, disk MB, note)
VARIANTS = (
    ("tiny", "quantized", 26, "Smallest and fastest; a little less accurate."),
    ("tiny", "float", 104, "Small and fast."),
    ("base", "quantized", 120, "More accurate than tiny, smaller than float."),
    ("base", "float", 239, "The most accurate."),
)
FREE_SHARE = 0.85  # keep some memory for the rest of the machine


def estimate_mb(disk_mb: int) -> int:
    return round(disk_mb * 1.5 + 300)


def variant_id(model: str, compute_type: str) -> str:
    return f"{model}|{compute_type}"


def fit(need_mb: int, hardware: dict[str, Any]) -> tuple[bool | None, str]:
    """(fits, reason). `None` = the free memory could not be read, so nothing is claimed."""
    free = hardware.get("ram_available_mb")
    if free is None:
        return None, "Free memory could not be read."
    ok = need_mb <= free * FREE_SHARE
    return ok, "" if ok else f"Needs about {need_mb} MB of memory; {free} MB is free."


def listing(hardware: dict[str, Any], current: tuple[str, str], downloaded: set[str]) -> list[dict[str, Any]]:
    """Every variant, smallest first. `current` is (model, compute_type) in use; it always fits, because the memory it
    holds already counts as used. `downloaded` holds the variant ids whose files are on disk."""
    rows = []
    for model, compute_type, disk_mb, note in VARIANTS:
        need, key = estimate_mb(disk_mb), variant_id(model, compute_type)
        in_use = (model, compute_type) == current
        fits, reason = (True, "") if in_use else fit(need, hardware)
        rows.append({"title": f"{model} · {compute_type} · CPU",
                     "detail": f"about {need} MB memory · {disk_mb} MB download{' · downloaded' if key in downloaded else ''} · English only · {note}",
                     "id": key, "model": model, "name": model, "compute_type": compute_type, "estimated_mb": need,
                     "disk_mb": disk_mb, "fits": fits, "reason": reason, "downloaded": key in downloaded,
                     "current": in_use, "note": note})
    return sorted(rows, key=lambda row: row["estimated_mb"])


def find(rows: list[dict[str, Any]], variant_id_: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["id"] == variant_id_), None)
