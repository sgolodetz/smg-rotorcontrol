import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from typing import cast, List, Optional, Tuple

from smg.navigation import PlanningToolkit
from smg.opengl import OpenGLUtil
from smg.rotory.drones import Drone, SimulatedDrone

from .drone_controller import DroneController


class LandingDroneController(DroneController):
    """A flight controller for a drone that tries to make the drone land."""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone, planning_toolkit: PlanningToolkit):
        """
        Construct a landing controller for a drone.

        :param drone:               The drone.
        :param planning_toolkit:    The planning toolkit (used for traversability checking).
        """
        super().__init__()
        self.__drone: Drone = drone
        self.__expected_end_pos: Optional[np.ndarray] = None
        self.__planning_toolkit: PlanningToolkit = planning_toolkit

    # PUBLIC METHODS

    def get_expected_end_pos(self) -> Optional[np.ndarray]:
        """
        Get the expected position of the drone once the controller has finished (if known).

        :return:    The expected position of the drone once the controller has finished, if known, or None otherwise.
        """
        # Get the expected start position of the drone (if known).
        expected_start_pos: Optional[np.ndarray] = self.get_expected_start_pos()

        # If the expected start position is known, and we haven't yet calculated the expected end position:
        if expected_start_pos is not None and self.__expected_end_pos is None:
            # Try to find a patch of flat ground below the current position of the drone.
            ground_vpos: Optional[np.ndarray] = self.__planning_toolkit.find_flat_ground_below(expected_start_pos)

            # If that succeeded, set the expected end position to one voxel above the centre of the patch.
            if ground_vpos is not None:
                resolution: float = self.__planning_toolkit.get_tree().get_resolution()
                self.__expected_end_pos = ground_vpos + np.array([0.0, -resolution, 0.0])

            # Otherwise, set the expected end position to be equal to the expected start position. One rationale
            # for this is that the controller will end up doing nothing, and so the drone won't move as a result.
            # A separate rationale is that we don't want the controller to keep trying to calculate the expected
            # end position repeatedly (which would happen if we didn't set it to something).
            else:
                self.__expected_end_pos = expected_start_pos

        return self.__expected_end_pos

    def get_expected_end_state(self) -> Optional[Drone.EState]:
        """
        Get the expected state of the drone once the controller has finished (if known).

        :return:    The expected state of the drone once the controller has finished, if known, or None otherwise.
        """
        return Drone.IDLE

    def has_finished(self) -> bool:
        """
        Get whether or not the controller has finished.

        :return:    True, if the controller has finished, or False otherwise.
        """
        # Note: If the drone state is unknown, we can't actually determine whether the controller has finished or not,
        #       so we simply assume that it hasn't.
        drone_state: Optional[Drone.EState] = self.__drone.get_state()
        return drone_state is not None and drone_state == Drone.IDLE

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

        # If the drone's known to be in the 'flying' state, ask it to land. Note that this will only be called once,
        # since once the drone has been asked to land, it will transition to the 'landing' state. Note also that if
        # the state of the drone can't be determined for some reason, this will do nothing.
        drone_state: Optional[Drone.EState] = self.__drone.get_state()
        if drone_state is not None and drone_state == Drone.FLYING:
            self.__drone.land()

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # Get the expected start and end positions for the controller. If either is unknown, early out.
        expected_start_pos: Optional[np.ndarray] = self.get_expected_start_pos()
        expected_end_pos: Optional[np.ndarray] = self.get_expected_end_pos()
        if expected_start_pos is None or expected_end_pos is None:
            return

        # Disable writing to the depth buffer. (This is to avoid the drone being blocked by the landing cone.)
        glDepthMask(False)

        # Enable blending.
        glEnable(GL_BLEND)
        glBlendColor(0.5, 0.5, 0.5, 0.5)
        glBlendFunc(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR)

        # Render a red, downwards-pointing cone to indicate the landing. Note that the base radius of 0.11m
        # is set to be ever so slightly larger than the radius of the spheres used to render new waypoints
        # on paths (see the 'traverse waypoints' drone controller). This avoids the depth fighting that
        # would occur if the same radius was used for both.
        glColor3f(1, 0, 0)
        OpenGLUtil.render_cylinder(expected_start_pos, expected_end_pos, 0.11, 0.0, slices=10)

        # Disable blending again.
        glDisable(GL_BLEND)

        # Enable writing to the depth buffer again.
        glDepthMask(True)
