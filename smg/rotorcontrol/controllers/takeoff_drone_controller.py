import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from typing import cast, List, Optional, Tuple

from smg.opengl import OpenGLUtil
from smg.rotory.drones import Drone, SimulatedDrone

from .drone_controller import DroneController


class TakeoffDroneController(DroneController):
    """A flight controller for a drone that tries to make the drone take off."""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        Construct a takeoff controller for a drone.

        :param drone:   The drone.
        """
        super().__init__()
        self.__drone: Drone = drone

    # PUBLIC METHODS

    def get_estimated_end_pos(self) -> Optional[np.ndarray]:
        """TODO"""
        # FIXME: Do this properly.
        return self.get_estimated_start_pos()

    def has_finished(self) -> bool:
        """
        Get whether or not the controller has finished.

        :return:    True, if the controller has finished, or False otherwise.
        """
        if type(self.__drone) is SimulatedDrone:
            simulated_drone: SimulatedDrone = cast(SimulatedDrone, self.__drone)
            return simulated_drone.get_state() == SimulatedDrone.FLYING
        else:
            # TODO: We still need to make this work for real drones.
            return False

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
        # Set the estimated start position to the current position of the drone if it's not already known.
        if self.get_estimated_start_pos() is None:
            self.set_estimated_start_pos(DroneController._extract_current_pos(tracker_c_t_i))

        # TODO: Comment here.
        if type(self.__drone) is SimulatedDrone:
            simulated_drone: SimulatedDrone = cast(SimulatedDrone, self.__drone)
            if simulated_drone.get_state() == SimulatedDrone.IDLE:
                self.__drone.takeoff()
        else:
            # TODO: We still need to make this work for real drones.
            pass

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        estimated_start_pos: Optional[np.ndarray] = self.get_estimated_start_pos()
        if estimated_start_pos is not None:
            glColor3f(0, 1, 0)
            OpenGLUtil.render_cylinder(
                estimated_start_pos, estimated_start_pos + np.array([0.0, -0.5, 0.0]), 0.1, 0.0, slices=10
            )
