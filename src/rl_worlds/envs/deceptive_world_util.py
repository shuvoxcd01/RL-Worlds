from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


class DeceptiveWorldUtil:
    """Visualization utilities for DeceptiveWorldEnv trajectories."""

    _SIZE_MAP = {"small": (4, 4), "medium": (5, 5), "large": (6, 6)}

    _COLOR_START = "#AED6F1"           # soft blue
    _COLOR_PSEUDO_TARGET = "#FAD7A0"   # warm amber
    _COLOR_REAL_TARGET = "#A9DFBF"     # soft green
    _COLOR_LEFTMOST = "#EBE8F8"        # faint lavender
    _COLOR_DEFAULT = "#FAFAFA"         # near-white
    _COLOR_GRID_EDGE = "#BDBDBD"

    @staticmethod
    def _reconstruct_states(
        trajectory: List[Tuple[int, int]], rows: int, cols: int
    ) -> List[int]:
        """Return the ordered list of states visited, including the terminal state.

        Parameters
        ----------
        trajectory:
            List of (prev_state, action) pairs as stored in info["trajectory"].
        rows, cols:
            Grid dimensions.
        """
        if not trajectory:
            return []

        states = [pair[0] for pair in trajectory]

        last_state, last_action = trajectory[-1]
        row, col = divmod(last_state, cols)

        if last_action == 0:      # up
            next_row, next_col = row - 1, col
        elif last_action == 1:    # right
            next_row, next_col = row, col + 1
        elif last_action == 2:    # down
            next_row, next_col = row + 1, col
        else:                     # left (3)
            next_row, next_col = row, col - 1

        next_row = max(0, min(rows - 1, next_row))
        next_col = max(0, min(cols - 1, next_col))
        states.append(next_row * cols + next_col)

        return states

    @staticmethod
    def visualize_trajectories(
        trajectories: List[List[Tuple[int, int]]],
        size: str = "small",
        labels: Optional[List[str]] = None,
        title: str = "DeceptiveWorld \u2014 Trajectory Visualization",
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
        show_state_ids: bool = True,
        arrow_lw: float = 2.0,
    ) -> plt.Figure:
        """Render one or more trajectories side-by-side on the DeceptiveWorld grid.

        Parameters
        ----------
        trajectories:
            List of trajectory lists, each a list of (prev_state, action) tuples
            as returned in info["trajectory"] from DeceptiveWorldEnv.step().
        size:
            Grid size — ``"small"`` (4×4), ``"medium"`` (5×5), or ``"large"`` (6×6).
        labels:
            Display names for each trajectory (used in the legend).  Defaults to
            ``["Trajectory 1", "Trajectory 2", ...]``.
        title:
            Figure title string.
        figsize:
            ``(width, height)`` in inches.  Auto-computed from grid size when ``None``.
        ax:
            Existing :class:`matplotlib.axes.Axes` to draw on.  A new figure is
            created when ``None``.
        show_state_ids:
            If ``True``, render the integer state index in the bottom of each cell.
        arrow_lw:
            Line-width for trajectory arrows.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if size not in DeceptiveWorldUtil._SIZE_MAP:
            raise ValueError(
                f"size must be one of {list(DeceptiveWorldUtil._SIZE_MAP.keys())}, got {size!r}"
            )

        rows, cols = DeceptiveWorldUtil._SIZE_MAP[size]
        n_states = rows * cols

        start_state = 0
        pseudo_target = (rows - 1) * cols
        real_target = n_states - 1
        leftmost_column = {r * cols for r in range(rows)}

        n_traj = len(trajectories)
        if labels is None:
            labels = [f"Trajectory {i + 1}" for i in range(n_traj)]

        if figsize is None:
            figsize = (max(6.5, cols * 1.6 + 3.5), max(5.0, rows * 1.6 + 1.0))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=110)
        else:
            fig = ax.get_figure()

        # Trajectory colour palette
        if n_traj <= 10:
            cmap = plt.get_cmap("tab10")
        else:
            cmap = plt.get_cmap("tab20")
        traj_colors = [cmap(i) for i in range(n_traj)]

        # Arc radii — fan paths apart on shared edges
        if n_traj <= 1:
            radii = [0.0]
        else:
            radii = list(np.linspace(-0.18, 0.18, n_traj))

        cell_size = 1.0
        padding = 0.08

        # ------------------------------------------------------------------ #
        # 1. Draw grid cells                                                   #
        # ------------------------------------------------------------------ #
        for state in range(n_states):
            r, c = divmod(state, cols)
            x = c * cell_size
            y = (rows - 1 - r) * cell_size   # row 0 at top

            if state == start_state:
                facecolor = DeceptiveWorldUtil._COLOR_START
            elif state == pseudo_target:
                facecolor = DeceptiveWorldUtil._COLOR_PSEUDO_TARGET
            elif state == real_target:
                facecolor = DeceptiveWorldUtil._COLOR_REAL_TARGET
            elif state in leftmost_column:
                facecolor = DeceptiveWorldUtil._COLOR_LEFTMOST
            else:
                facecolor = DeceptiveWorldUtil._COLOR_DEFAULT

            patch = FancyBboxPatch(
                (x + padding, y + padding),
                cell_size - 2 * padding,
                cell_size - 2 * padding,
                boxstyle="round,pad=0.02",
                facecolor=facecolor,
                edgecolor=DeceptiveWorldUtil._COLOR_GRID_EDGE,
                linewidth=1.2,
                zorder=1,
            )
            ax.add_patch(patch)

            # State ID — bottom of cell
            if show_state_ids:
                ax.text(
                    x + cell_size / 2,
                    y + 0.22,
                    str(state),
                    ha="center", va="center",
                    fontsize=8, color="#AAAAAA",
                    zorder=2,
                )

        # ------------------------------------------------------------------ #
        # 2. Cell annotations (icon + text label)                              #
        # ------------------------------------------------------------------ #
        annotations = {
            start_state: ("S", "Start", "#2874A6"),
            pseudo_target: ("\u2691", "Pseudo\nTarget", "#A04000"),
            real_target: ("\u2605", "Real\nTarget", "#1E8449"),
        }
        for state, (icon, label_text, color) in annotations.items():
            r, c = divmod(state, cols)
            x = c * cell_size
            y = (rows - 1 - r) * cell_size
            ax.text(
                x + cell_size / 2, y + cell_size * 0.65,
                icon,
                ha="center", va="center",
                fontsize=13, fontweight="bold", color=color,
                zorder=2,
            )
            ax.text(
                x + cell_size / 2, y + cell_size * 0.42,
                label_text,
                ha="center", va="center",
                fontsize=5.5, color=color,
                zorder=2,
            )

        # ------------------------------------------------------------------ #
        # 3. Draw trajectories                                                  #
        # ------------------------------------------------------------------ #
        action_offsets = {0: (0, 0.32), 1: (0.32, 0), 2: (0, -0.32), 3: (-0.32, 0)}

        for traj_idx, (trajectory, color, rad) in enumerate(
            zip(trajectories, traj_colors, radii)
        ):
            states = DeceptiveWorldUtil._reconstruct_states(trajectory, rows, cols)
            if not states:
                continue

            # Origin dot at step 0
            r0, c0 = divmod(states[0], cols)
            x0 = c0 * cell_size + cell_size / 2
            y0 = (rows - 1 - r0) * cell_size + cell_size / 2
            ax.plot(x0, y0, "o", color=color, markersize=7, zorder=5)

            for step_idx in range(len(states) - 1):
                s_from = states[step_idx]
                s_to = states[step_idx + 1]

                rf, cf = divmod(s_from, cols)
                rt, ct = divmod(s_to, cols)

                xf = cf * cell_size + cell_size / 2
                yf = (rows - 1 - rf) * cell_size + cell_size / 2
                xt = ct * cell_size + cell_size / 2
                yt = (rows - 1 - rt) * cell_size + cell_size / 2

                if s_from == s_to:
                    # Wall hit — draw ✕ near the blocked edge
                    action = trajectory[step_idx][1]
                    dx, dy = action_offsets.get(action, (0, 0))
                    ax.text(
                        xf + dx, yf + dy, "\u2715",
                        ha="center", va="center",
                        fontsize=9, color=color,
                        zorder=5,
                    )
                else:
                    ax.annotate(
                        "",
                        xy=(xt, yt),
                        xytext=(xf, yf),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=arrow_lw,
                            connectionstyle=f"arc3,rad={rad}",
                        ),
                        zorder=4,
                    )

            # Terminal star at final state
            rf, cf = divmod(states[-1], cols)
            xfin = cf * cell_size + cell_size / 2
            yfin = (rows - 1 - rf) * cell_size + cell_size / 2
            ax.plot(xfin, yfin, "*", color=color, markersize=12, zorder=5)

        # ------------------------------------------------------------------ #
        # 4. Two legends outside the axes                                       #
        # ------------------------------------------------------------------ #
        traj_handles = [
            mpatches.Patch(facecolor=traj_colors[i], label=labels[i])
            for i in range(n_traj)
        ]
        cell_handles = [
            mpatches.Patch(
                facecolor=DeceptiveWorldUtil._COLOR_START,
                edgecolor=DeceptiveWorldUtil._COLOR_GRID_EDGE,
                label="Start",
            ),
            mpatches.Patch(
                facecolor=DeceptiveWorldUtil._COLOR_PSEUDO_TARGET,
                edgecolor=DeceptiveWorldUtil._COLOR_GRID_EDGE,
                label="Pseudo Target",
            ),
            mpatches.Patch(
                facecolor=DeceptiveWorldUtil._COLOR_REAL_TARGET,
                edgecolor=DeceptiveWorldUtil._COLOR_GRID_EDGE,
                label="Real Target",
            ),
            mpatches.Patch(
                facecolor=DeceptiveWorldUtil._COLOR_LEFTMOST,
                edgecolor=DeceptiveWorldUtil._COLOR_GRID_EDGE,
                label="Leftmost Col (+10)",
            ),
        ]

        legend_traj = ax.legend(
            handles=traj_handles,
            title="Trajectories",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            framealpha=0.9,
            fontsize=8,
            title_fontsize=9,
        )
        ax.add_artist(legend_traj)
        ax.legend(
            handles=cell_handles,
            title="Cell Types",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            borderaxespad=0,
            framealpha=0.9,
            fontsize=8,
            title_fontsize=9,
        )

        # ------------------------------------------------------------------ #
        # 5. Axis styling                                                        #
        # ------------------------------------------------------------------ #
        ax.set_xlim(-0.05, cols * cell_size + 0.05)
        ax.set_ylim(-0.05, rows * cell_size + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        fig.tight_layout()
        return fig
