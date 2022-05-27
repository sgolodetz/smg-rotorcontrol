import collections
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import vg

from OpenGL.GL import *
from typing import Any, Dict, List, Optional, Tuple

from smg.opengl import OpenGLUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController


class RateCalibrationDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        TODO

        :param drone:   TODO
        """
        super().__init__()

        self.__drone: Drone = drone
        self.__durations: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        self.__maxs: Optional[np.ndarray] = None
        self.__mins: Optional[np.ndarray] = None
        self.__origin: Optional[np.ndarray] = None
        self.__rate_idx: int = 0
        self.__rates: np.ndarray = np.linspace(0.3, 0.1, 5, endpoint=True)
        self.__stage: int = 0
        self.__which: str = "forward"

    # PUBLIC METHODS

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
        # Make a copy of the keyword arguments that have been passed to the method, so that they can later be
        # forwarded on to the subsidiary iterate functions.
        kwargs: Dict[str, Any] = {key: value for key, value in locals().items() if key != "self"}

        # FIXME: Make sure that tracker_c_t_i is not None.

        # TODO: Comment here.
        if self.__which == "":
            self.__drone.stop()
            return

        # Convert the drone's pose to a camera for easier manipulation.
        cam: SimpleCamera = CameraPoseConverter.pose_to_camera(tracker_c_t_i)

        # TODO: Comment here.
        if self.__origin is None:
            self.__origin = cam.p().copy()
            self.__mins = self.__origin - np.array([0.5, 1, 0])
            self.__maxs = self.__origin + np.array([0.5, 0, 1])

        if self.__stage == 0:
            # FIXME: This is a duplicate of the code in TraversePathDroneController - factor it out.
            # Determine the current orientation of the drone in the horizontal plane.
            current_n: np.ndarray = vg.normalize(np.array([cam.n()[0], 0, cam.n()[2]]))

            # Determine the target orientation of the drone in the horizontal plane.
            if self.__which == "forward" or self.__which == "up":
                target_n: np.ndarray = np.array([0, 0, 1])
            else:
                target_n: np.ndarray = np.array([-1, 0, 0])

            # Determine whether the drone needs to turn left or right to achieve the target orientation.
            # (Note that y points downwards in our coordinate system!)
            cp: np.ndarray = np.cross(current_n, target_n)
            sign: int = 1 if np.dot(cp, np.array([0, -1, 0])) >= 0 else -1

            # Determine the angle by which the drone needs to turn to achieve the target orientation.
            angle: float = sign * np.arccos(np.clip(np.dot(current_n, target_n), -1.0, 1.0))

            if np.fabs(angle) > 2.0 * np.pi / 180:
                # Determine an appropriate turn rate for the drone.
                turn_rate: float = np.clip(-angle / (np.pi / 2), -1.0, 1.0)

                # TODO: Comment here.
                self.__drone.move_forward(0.0)
                self.__drone.move_right(0.0)
                self.__drone.move_up(0.0)
                self.__drone.turn(turn_rate)
            else:
                # TODO
                self.__stage = 1
                self.__drone.stop()

        else:
            sign: int = 1 if self.__stage == 1 else -1
            signed_rate: float = sign * self.__rates[self.__rate_idx]

            if self.__which == "forward":
                if self.__stage == 1 and cam.p()[2] >= self.__maxs[2]:
                    self.__stage = 2
                elif self.__stage == 2 and cam.p()[2] <= self.__mins[2]:
                    self.__drone.stop()
                    self.__stage = 0

                    if self.__rate_idx + 1 < len(self.__rates):
                        self.__rate_idx += 1
                    else:
                        self.__rate_idx = 0
                        self.__which = "right"
                else:
                    self.__drone.move_forward(signed_rate)
            elif self.__which == "right":
                if self.__stage == 1 and cam.p()[2] >= self.__maxs[2]:
                    self.__stage = 2
                elif self.__stage == 2 and cam.p()[2] <= self.__mins[2]:
                    self.__drone.stop()
                    self.__stage = 0

                    if self.__rate_idx + 1 < len(self.__rates):
                        self.__rate_idx += 1
                    else:
                        self.__rate_idx = 0
                        self.__which = "up"
                else:
                    self.__drone.move_right(signed_rate)
            elif self.__which == "up":
                if self.__stage == 1 and cam.p()[1] <= self.__mins[1]:
                    self.__stage = 2
                elif self.__stage == 2 and cam.p()[1] >= self.__maxs[1]:
                    self.__drone.stop()
                    self.__stage = 0

                    if self.__rate_idx + 1 < len(self.__rates):
                        self.__rate_idx += 1
                    else:
                        self.__rate_idx = 0
                        self.__which = ""
                else:
                    self.__drone.move_up(signed_rate)

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        if self.__origin is None:
            return

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor3f(1, 0, 0)
        OpenGLUtil.render_aabb(self.__mins, self.__maxs)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
