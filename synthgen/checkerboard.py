from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

from synthgen.config import CheckerboardConfig
from synthgen.io import get_chessboard_fiducial_point


class Checkerboard:
    """Checkerboard class defining number of points on checkerboard and physical dimension"""

    def __init__(
        self,
        config: CheckerboardConfig,
        origin: np.array = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    ) -> None:
        self.config: CheckerboardConfig = config
        self.points: np.array = self.generate_checkerboard_points()
        self.origin: np.array = origin

        fiducial, _ = get_chessboard_fiducial_point(self.config.fiducial_file)
        self.fiducials: np.array = np.vstack([v for k, v in fiducial.items()])

    def __repr__(self) -> str:
        return f"Checkerboard(origin={str(self.origin)}, points={str(self.points)})"

    def generate_checkerboard_points(self):
        """Generate points on checkerboard given number of points in each dimension nx, ny and defining the square length"""

        nx = self.config.num_points_x
        ny = self.config.num_points_y
        square_length_m = self.config.square_length_m

        ids_x_inner = np.arange(0, nx - 1)
        ids_y_inner = np.arange(0, ny - 1)

        pts_x_inner = square_length_m + ids_x_inner * square_length_m
        pts_y_inner = square_length_m + ids_y_inner * square_length_m

        yy_inner, xx_inner = np.meshgrid(
            pts_y_inner, pts_x_inner
        )  # by convention use y as first axis

        positions_inner = np.vstack(
            [xx_inner.ravel(), yy_inner.ravel(), np.zeros_like(yy_inner.ravel())]
        )

        self.num_points = positions_inner.shape[-1]

        return positions_inner.T

    def plot(self):
        """Plot checkerboard points"""
        plt.scatter(self.points[:, 0], self.points[:, 1], color="black", marker="s")
        txt_margin = 0.004
        plt.scatter([0], [0], color="purple", label="checkerboard origin")
        [
            plt.text(
                self.points[el, 0] + txt_margin,
                self.points[el, 1] + txt_margin,
                s=str(el),
            )
            for el in range(self.num_points)
        ]
        plt.arrow(
            0,
            0,
            0.06,
            0,
            color="red",
            width=0.001,
            head_width=0.01,
            length_includes_head=True,
        )
        plt.arrow(
            0,
            0,
            0.0,
            0.06,
            color="green",
            width=0.001,
            head_width=0.01,
            length_includes_head=True,
        )
        plt.suptitle("Checkerboard Points")
        plt.gca().set_aspect("equal")
        plt.legend()
