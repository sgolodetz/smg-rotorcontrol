import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from typing import List, Optional, Tuple

from smg.opengl import OpenGLUtil
from smg.rotory.drones import Drone

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

    def get_expected_end_pos(self) -> Optional[np.ndarray]:
        """
        Get the expected position of the drone once the controller has finished (if known).

        :return:    The expected position of the drone once the controller has finished, if known, or None otherwise.
        """
        expected_start_pos: Optional[np.ndarray] = self.get_expected_start_pos()
        if expected_start_pos is None:
            return None

        expected_takeoff_height: Optional[float] = self.__drone.get_expected_takeoff_height()
        if expected_takeoff_height is None:
            expected_takeoff_height = 1.0

        return expected_start_pos + np.array([0.0, -expected_takeoff_height, 0.0])

    def get_expected_end_state(self) -> Optional[Drone.EState]:
        """
        Get the expected state of the drone once the controller has finished (if known).

        :return:    The expected state of the drone once the controller has finished, if known, or None otherwise.
        """
        return Drone.FLYING

    def has_finished(self) -> bool:
        """
        Get whether or not the controller has finished.

        :return:    True, if the controller has finished, or False otherwise.
        """
        # Note: If the drone state is unknown, we can't actually determine whether the controller has finished or not,
        #       so we simply assume that it hasn't.
        drone_state: Optional[Drone.EState] = self.__drone.get_state()
        return drone_state is not None and drone_state == Drone.FLYING

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
        :param tracker_c_t_i:       The 6D pose of the drone, expressed as a 4x4 matrix representing a transformation
                                    from initial camera space to current camera space, as estimated by any tracker that
                                    is running (optional). Note that if the tracker is monocular, the transformation is
                                    unlikely to be scale-correct.
        """
        # Set the estimated start position to the current position of the drone if it's not already known.
        if self.get_expected_start_pos() is None:
            self.set_expected_start_pos(DroneController._extract_current_pos(tracker_c_t_i))

        # If the drone's known to be in the 'idle' state, ask it to take off. Note that this will only be called once,
        # since once the drone has been asked to take off, it will transition to the 'taking off' state. Note also that
        # if the state of the drone can't be determined for some reason, this will do nothing.
        drone_state: Optional[Drone.EState] = self.__drone.get_state()
        if True:  # drone_state is not None and drone_state == Drone.IDLE:
            self.__drone.takeoff()

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # Get the expected start and end positions for the controller. If either is unknown, early out.
        expected_start_pos: Optional[np.ndarray] = self.get_expected_start_pos()
        expected_end_pos: Optional[np.ndarray] = self.get_expected_end_pos()
        if expected_start_pos is None or expected_end_pos is None:
            return

        # Disable writing to the depth buffer. (This is to avoid the drone being blocked by the takeoff cone.)
        glDepthMask(False)

        # Enable blending.
        glEnable(GL_BLEND)
        glBlendColor(0.5, 0.5, 0.5, 0.5)
        glBlendFunc(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR)

        # Render a green, upwards-pointing cone to indicate the takeoff. Note that the base radius of 0.11m is set
        # to be ever so slightly larger than the radius of the spheres used to render new waypoints on paths (see
        # the 'traverse waypoints' drone controller). This avoids the depth fighting that would occur if the same
        # radius was used for both.
        glColor3f(0, 1, 0)
        OpenGLUtil.render_cylinder(expected_start_pos, expected_end_pos, 0.11, 0.0, slices=10)

        # Disable blending again.
        glDisable(GL_BLEND)

        # Enable writing to the depth buffer again.
        glDepthMask(True)
