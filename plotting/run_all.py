#!/usr/bin/env python3
from __future__ import annotations

from plot_mean_total_steps_bar import main as mean_total_steps_bar_main
from plot_observes_mega import main as observes_mega_main
from plot_pooled_observe_steps import main as pooled_observe_main
from plot_pooled_total_steps import main as pooled_total_main
from plot_total_steps_mega import main as total_steps_mega_main


def main() -> int:
    pooled_observe_main()
    pooled_total_main()
    mean_total_steps_bar_main()
    total_steps_mega_main()
    observes_mega_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
