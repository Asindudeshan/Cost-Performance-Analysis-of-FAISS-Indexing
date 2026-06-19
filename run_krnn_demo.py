import numpy as np
from research_report import (
    resource_aware_search,
    get_system_stats,
    print_system_stats,
    set_ram_threshold,
    set_cpu_threshold,
    set_swap_threshold,
    set_min_free_ram,
    set_disk_io_threshold,
)

# ── 1. Show current system snapshot ──────────────────────────────────────────
print("=" * 60)
print(" SYSTEM RESOURCE SNAPSHOT")
print("=" * 60)
stats = get_system_stats()
print_system_stats(stats)

# ── 2. (Optional) Override any threshold before searching ────────────────────
# set_ram_threshold(85.0)       # lower RAM limit  → stricter routing
# set_cpu_threshold(70.0)       # lower CPU limit  → stricter routing
# set_swap_threshold(30.0)      # lower Swap limit → stricter routing
# set_min_free_ram(1024)        # require at least 1 GB free RAM
# set_disk_io_threshold(60.0)   # lower disk I/O limit → stricter routing

# ── 3. Run resource-aware KNN search ─────────────────────────────────────────
print("\n" + "=" * 60)
print(" RESOURCE-AWARE KNN SEARCH  (k=5)")
print("=" * 60)

dim   = 512
query = np.random.random((dim,)).astype("float32")

distances, ids = resource_aware_search(
    query_vector   = query,
    k              = 5,
    # All thresholds can be overridden per-call:
    ram_threshold  = 95.0,    # RAM  %  (default 95 %)
    cpu_threshold  = 80.0,    # CPU  %  (default 80 %)
    swap_threshold = 50.0,    # Swap %  (default 50 %)
    min_free_ram   = 512,     # Free RAM MB (default 512 MB)
    disk_threshold = 80.0,    # Disk I/O % (default 80 %)
)

print("\nTop-5 IDs   :", ids[0])
print("Distances   :", distances[0])
