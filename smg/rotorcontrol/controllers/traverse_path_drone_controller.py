import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import vg

from typing import List, Optional, Tuple

from smg.navigation import Path
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController


class TraversePathDroneController(DroneController):
    """A flight controller for a drone that tries to traverse a planned path."""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        Construct a flight controller for a drone that tries to traverse a planned path.

        :param drone:   The drone.
        """
        super().__init__()

        self.__drone: Drone = drone
        self.__path: Optional[Path] = None

    # PUBLIC METHODS

    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
                image: np.ndarray, image_timestamp: Optional[float] = None,
                intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        .. note::
            This controller (i) requires the tracker poses to be passed in, and (ii) requires that they be
            scale-correct. We explicitly check (i). We can't check (ii), so client code is responsible for
            correct use.

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
        # If no tracker pose has been passed in, stop the drone and early out.
        if tracker_c_t_i is None:
            self.__drone.stop()
            return

        # Extract the current position of the drone from the tracker pose provided.
        current_pos: np.ndarray = DroneController._extract_current_pos(tracker_c_t_i)

        # A flag indicating whether or not the drone should stop moving. This will be set to False if any reason
        # is found for the drone to continue moving.
        stop_drone: bool = True

        # If there's a current path, try to follow it.
        if self.__path is not None:
            # First compute a vector from the drone's current position to the next waypoint on the path.
            offset: np.ndarray = self.__path[1].position - current_pos

            # Provided we're far enough from the next waypoint for the vector towards it to be normalised:
            offset_length: float = np.linalg.norm(offset)
            if offset_length >= 1e-4:
                # Convert the drone's pose to a camera for easier manipulation.
                cam: SimpleCamera = CameraPoseConverter.pose_to_camera(tracker_c_t_i)

                # Project the vector to the next waypoint into the horizontal plane.
                horizontal_offset: np.ndarray = np.array([offset[0], 0, offset[2]])

                # If we're far enough horizontally from the next waypoint to turn the drone before we get there:
                horizontal_offset_length: float = np.linalg.norm(horizontal_offset)
                if horizontal_offset_length >= 0.1:
                    # Determine the current orientation of the drone in the horizontal plane.
                    current_n: np.ndarray = vg.normalize(np.array([cam.n()[0], 0, cam.n()[2]]))

                    # Determine the target orientation of the drone in the horizontal plane.
                    target_n: np.ndarray = vg.normalize(horizontal_offset)

                    # Determine whether the drone needs to turn left or right to achieve the target orientation.
                    # (Note that y points downwards in our coordinate system!)
                    cp: np.ndarray = np.cross(current_n, target_n)
                    sign: int = 1 if np.dot(cp, np.array([0, -1, 0])) >= 0 else -1

                    # Determine the angle by which the drone needs to turn to achieve the target orientation.
                    angle: float = sign * np.arccos(np.clip(np.dot(current_n, target_n), -1.0, 1.0))

                    # Determine an appropriate turn rate for the drone.
                    turn_rate: float = np.clip(-angle / (np.pi / 2), -1.0, 1.0)

                # Otherwise, set both the angle by which the drone needs to turn and the turn rate to zero.
                else:
                    angle: float = 0.0
                    turn_rate: float = 0.0

                # Set the drone's turn rate.
                self.__drone.turn(turn_rate)

                # Determine the linear rates at which the drone should in principle move in each of the three axes.
                speed: float = 0.3
                normalized_offset: np.ndarray = offset / offset_length
                forward_rate: float = vg.scalar_projection(normalized_offset, cam.n()) * speed
                right_rate: float = vg.scalar_projection(normalized_offset, -cam.u()) * speed
                up_rate: float = vg.scalar_projection(normalized_offset, cam.v()) * speed

                # If the drone's current orientation is within a reasonable angle of its target orientation,
                # or alternatively if the drone is not currently turning, set the calculated linear rates.
                if np.fabs(angle) * 180 / np.pi <= 30.0 or turn_rate == 0.0:
                    self.__drone.move_forward(forward_rate)
                    self.__drone.move_right(right_rate)
                    self.__drone.move_up(up_rate)

                # Otherwise, suppress the drone's linear movement.
                else:
                    self.__drone.move_forward(0.0)
                    self.__drone.move_right(0.0)
                    self.__drone.move_up(0.0)

                # Set the flag that will cause the drone to be stopped below to False, since we have just set the
                # rates we want manually.
                stop_drone = False

        # If the drone should stop moving, stop it.
        if stop_drone:
            self.__drone.stop()

    def set_path(self, path: Optional[Path]) -> None:
        """
        Set the path that the controller should try to make the drone traverse.

        .. note::
            The path must start at the current position of the drone.

        :param path:    The path that the controller should try to make the drone traverse.
        """
        self.__path = path
