import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.rotory.drones import Drone

from .drone_controller import DroneController


# FIXME: Temporary code.
def rad_per_s_to_rate(rad_per_s: float) -> float:
    if rad_per_s <= 0.526913819:
        return (-0.5211 + np.sqrt(0.2937691266672 + 5.8512 * rad_per_s)) / 2.9256
    else:
        return (rad_per_s + 0.306576181) / 1.8522


class PPNDroneController(DroneController):
    """
    A flight controller for a drone that uses pure proportional navigation to fly towards a target.

    .. note::
        The target can be changed on-the-fly as desired.
    """

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        Construct a pure proportional navigation flight controller for a drone.

        :param drone:   The drone.
        """
        super().__init__()

        self.__drone: Drone = drone
        self.__rd_old: Optional[np.ndarray] = None
        self.__speed: float = 0.5
        self.__t_old: Optional[float] = None
        self.__target_pos: Optional[np.ndarray] = None

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
        # If no tracker pose has been passed in, raise an exception and early out.
        if tracker_c_t_i is None:
            raise RuntimeError("Error: Tracker poses must be provided when using 'PPN' control")

        # TODO: Comment here.
        if self.__target_pos is None:
            self.__drone.stop()
            return

        # TODO: Comment here.
        drone_pos: np.ndarray = DroneController._extract_current_pos(tracker_c_t_i)

        # TODO: Comment here.
        rd: np.ndarray = self.__target_pos - drone_pos
        rd[1] = 0.0

        # TODO: Comment here.
        t: float = timer()

        # TODO: Comment here.
        if self.__rd_old is not None:
            cp: np.ndarray = np.cross(self.__rd_old, rd)
            sign: int = -1 if np.dot(cp, np.array([0, -1, 0])) >= 0 else 1
            rad_per_s: float = \
                np.linalg.norm(cp) / (np.linalg.norm(self.__rd_old) * np.linalg.norm(rd)) / (t - self.__t_old)
            rate: float = sign * rad_per_s_to_rate(rad_per_s)
            n: float = 3.0
            print(self.__target_pos, drone_pos, rd, self.__rd_old, cp, sign, rate)
            self.__drone.move_forward(self.__speed)
            self.__drone.turn(n * rate)

        self.__t_old = t
        self.__rd_old = rd.copy()

    def reset(self) -> None:
        """Reset the controller."""
        self.__rd_old = None
        self.__t_old = None
        self.__target_pos = None

    def set_target_pos(self, target_pos: Optional[np.ndarray]) -> None:
        """
        Set the target position for the drone.

        :param target_pos:  The target position for the drone (optional).
        """
        self.__target_pos = target_pos.copy() if target_pos is not None else None
