import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class DroneController(ABC):
    """A flight controller for a drone."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
                image: np.ndarray, image_timestamp: Optional[float] = None,
                intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        :param altitude:            The most recent altitude (in m) for the drone, as measured by any height sensor
                                    it is carrying (optional).
        :param events:              An optional list of PyGame events that have happened since the last iteration.
        :param image:               The most recent image from the drone.
        :param image_timestamp:     The timestamp of the most recent image from the drone (optional).
        :param intrinsics:          The intrinsics of the drone's camera.
        :param tracker_c_t_i:       A transformation from initial camera space to current camera space, as estimated
                                    by any tracker that's running (optional). Note that if the tracker is a monocular
                                    one, the transformation will be non-metric.
        """
        pass

    # PUBLIC METHODS

    # noinspection PyMethodMayBeStatic
    def has_finished(self) -> bool:
        """
        Get whether or not the controller has finished.

        :return:    True, if the controller has finished, or False otherwise.
        """
        return False

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        pass

    def terminate(self) -> None:
        """Tell the controller to terminate."""
        pass
