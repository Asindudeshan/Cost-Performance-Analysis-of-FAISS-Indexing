# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
virtual_resource_simulator.py
─────────────────────────────
Simulates different system-resource scenarios (RAM, CPU, Swap,
Free-RAM, Disk-I/O) WITHOUT needing real hardware pressure.

Uses unittest.mock.patch to inject fake psutil values so the
resource_aware_search router can be tested under any condition.
"""

from unittest.mock import patch, MagicMock
import numpy as np
import faiss
import os

# ── local import ─────────────────────────────────────────────────────────────
from research_report import resource_aware_search

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a fake psutil environment
# ─────────────────────────────────────────────────────────────────────────────
def _make_vm(used_pct: float, free_mb: float):
    """Return a mock virtual_memory() object."""
    vm = MagicMock()
    vm.percent   = used_pct
    vm.available = free_mb * 1024 * 1024   # bytes
    return vm

def _make_swap(used_pct: float):
    """Return a mock swap_memory() object."""
    sw = MagicMock()
    sw.percent = used_pct
    return sw

def _make_disk_io(busy_pct: float):
    """
    Return two mock disk_io_counters() snapshots 100 ms apart
    such that the calculated busy % equals *busy_pct*.
    """
    # busy_ms per 100 ms window  →  busy_pct = (busy_ms / 100) * 100
    busy_ms = busy_pct   # same numeric value
    io1 = MagicMock(); io1.read_time = 0;        io1.write_time = 0
    io2 = MagicMock(); io2.read_time = busy_ms;  io2.write_time = 0
    return [io1, io2]   # first call returns io1, second returns io2


# ─────────────────────────────────────────────────────────────────────────────
# Core: run one simulated scenario
# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(
    name: str,
    ram_pct:    float = 50.0,
    cpu_pct:    float = 30.0,
    swap_pct:   float = 10.0,
    free_mb:    float = 2048.0,
    disk_pct:   float = 10.0,
    # thresholds (same defaults as research_report.py)
    ram_threshold:  float = 95.0,
    cpu_threshold:  float = 80.0,
    swap_threshold: float = 50.0,
    min_free_ram:   float = 512.0,
    disk_threshold: float = 80.0,
):
    """
    Inject fake resource values and run resource_aware_search.
    Returns the string 'IVF' or 'HNSW' depending on which index was chosen.
    """
    print("\n" + "=" * 65)
    print(f"  SCENARIO : {name}")
    print("-" * 65)
    print(f"  RAM={ram_pct}%  CPU={cpu_pct}%  Swap={swap_pct}%  "
          f"FreeRAM={free_mb}MB  DiskIO={disk_pct}%")
    print("─" * 65)

    disk_io_side_effects = _make_disk_io(disk_pct)

    # We also need to mock faiss.read_index so it doesn't need real .faiss files
    dummy_index = MagicMock()
    dummy_index.search.return_value = (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype="float32"),
        np.array([[0, 1, 2, 3, 4]],           dtype="int64"),
    )

    chosen = {"index": None}
    original_read_index = faiss.read_index

    def mock_read_index(path):
        chosen["index"] = "IVF" if "ivf" in path.lower() else "HNSW"
        return dummy_index

    query = np.random.random((512,)).astype("float32")

    with (
        patch("research_report.psutil.virtual_memory",
              return_value=_make_vm(ram_pct, free_mb)),
        patch("research_report.psutil.cpu_percent",
              return_value=cpu_pct),
        patch("research_report.psutil.swap_memory",
              return_value=_make_swap(swap_pct)),
        patch("research_report.psutil.disk_io_counters",
              side_effect=disk_io_side_effects),
        patch("research_report.faiss.read_index",
              side_effect=mock_read_index),
    ):
        resource_aware_search(
            query,
            k              = 5,
            ram_threshold  = ram_threshold,
            cpu_threshold  = cpu_threshold,
            swap_threshold = swap_threshold,
            min_free_ram   = min_free_ram,
            disk_threshold = disk_threshold,
        )

    result = chosen["index"] or "HNSW"   # fallback if mock didn't fire
    emoji  = "[IVF]  safe mode  " if result == "IVF" else "[HNSW] high recall"
    print(f"\n  --> ROUTER DECISION: {emoji}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scenario suite
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    # name,                    ram   cpu  swap  freeRAM  disk
    ("✅ All Healthy",          50.0, 30.0, 10.0, 2048.0,  5.0),
    ("🔴 RAM too high",         96.0, 30.0, 10.0, 2048.0,  5.0),
    ("🔴 CPU too high",         50.0, 85.0, 10.0, 2048.0,  5.0),
    ("🔴 Swap too high",        50.0, 30.0, 55.0, 2048.0,  5.0),
    ("🔴 Free RAM too low",     50.0, 30.0, 10.0,  256.0,  5.0),
    ("🔴 Disk I/O too high",    50.0, 30.0, 10.0, 2048.0, 85.0),
    ("🔴 RAM + CPU both high",  97.0, 90.0, 10.0, 2048.0,  5.0),
    ("🔴 All metrics critical", 98.0, 95.0, 70.0,  128.0, 95.0),
    ("✅ Edge: exactly at limit",94.9, 79.9, 49.9, 512.1,  79.9),
]


def main():
    print("\n" + "#" * 65)
    print("  VIRTUAL RESOURCE SIMULATOR -- KNN Routing Test Suite")
    print("#" * 65)

    results = {}
    for row in SCENARIOS:
        name, ram, cpu, swap, free, disk = row
        decision = run_scenario(
            name      = name,
            ram_pct   = ram,
            cpu_pct   = cpu,
            swap_pct  = swap,
            free_mb   = free,
            disk_pct  = disk,
        )
        results[name] = decision

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  SIMULATION SUMMARY")
    print("=" * 65)
    print(f"  {'Scenario':<35} {'Decision'}")
    print("-" * 65)
    for name, dec in results.items():
        badge = "[HNSW] high recall" if dec == "HNSW" else "[IVF]  safe mode  "
        print(f"  {name:<40} {badge}")
    print("=" * 65)

    ivf_count  = sum(1 for v in results.values() if v == "IVF")
    hnsw_count = sum(1 for v in results.values() if v == "HNSW")
    print(f"  IVF  (safe mode)   selected : {ivf_count} / {len(results)} scenarios")
    print(f"  HNSW (high recall) selected : {hnsw_count} / {len(results)} scenarios")
    print("=" * 65)


if __name__ == "__main__":
    main()
