import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.rotory.drones import Drone

from .drone_controller import DroneController


# FIXME: Temporary code (Tello version).
# def rad_per_s_to_rate(rad_per_s: float) -> float:
#     if rad_per_s <= 0.526913819:
#         return (-0.5211 + np.sqrt(0.2937691266672 + 5.8512 * rad_per_s)) / 2.9256
#     else:
#         return (rad_per_s + 0.306576181) / 1.8522

# FIXME: Temporary code (simulated drone version).
def rad_per_s_to_rate(rad_per_s: float, t_diff: float) -> float:
    angular_gain: float = 0.02
    return rad_per_s * t_diff / angular_gain


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
        self.__drone_pos: Optional[np.ndarray] = None
        self.__rd_old: Optional[np.ndarray] = None
        self.__speed: float = 0.5
        self.__t_old: Optional[float] = None
        self.__target_pos: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def get_expected_end_pos(self) -> Optional[np.ndarray]:
        """
        Get the expected position of the drone once the controller has finished (if known).

        :return:    The expected position of the drone once the controller has finished, if known, or None otherwise.
        """
        # FIXME: Do this properly.
        expected_start_pos: Optional[np.ndarray] = self.get_expected_start_pos()
        if expected_start_pos is not None and self.__target_pos is not None:
            return np.array([self.__target_pos[0], expected_start_pos[1], self.__target_pos[2]])
        else:
            return None

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
        # FIXME: Do this properly.
        if self.__target_pos is None:
            return True
        elif self.__drone_pos is not None:
            rd: np.ndarray = self.__target_pos - self.__drone_pos
            rd[1] = 0.0
            return np.linalg.norm(rd) < 0.025
        else:
            return False

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
        if self.has_finished():
            self.__drone.stop()
            return

        # TODO: Comment here.
        self.__drone_pos = DroneController._extract_current_pos(tracker_c_t_i)

        # Set the estimated start position to the current position of the drone if it's not already known.
        if self.get_expected_start_pos() is None:
            self.set_expected_start_pos(self.__drone_pos.copy())

        # TODO: Comment here.
        rd: np.ndarray = self.__target_pos - self.__drone_pos
        rd[1] = 0.0

        # TODO: Comment here.
        t: float = timer()

        # TODO: Comment here.
        if self.__rd_old is not None:
            cp: np.ndarray = np.cross(self.__rd_old, rd)
            sign: int = -1 if np.dot(cp, np.array([0, -1, 0])) >= 0 else 1
            rad_per_s: float = \
                np.linalg.norm(cp) / (np.linalg.norm(self.__rd_old) * np.linalg.norm(rd)) / (t - self.__t_old)
            rate: float = sign * rad_per_s_to_rate(rad_per_s, t - self.__t_old)
            n: float = 1.5
            # print(self.__target_pos, self.__drone_pos, rd, self.__rd_old, cp, sign, rad_per_s, t - self.__t_old, rate)
            self.__drone.move_forward(self.__speed)
            self.__drone.turn(n * rate)

        self.__t_old = t
        self.__rd_old = rd.copy()

    def reset(self) -> None:
        """Reset the controller."""
        self.__drone_pos = None
        self.__rd_old = None
        self.__t_old = None
        self.__target_pos = None

    def set_target_pos(self, target_pos: Optional[np.ndarray]) -> None:
        """
        Set the target position for the drone.

        :param target_pos:  The target position for the drone (optional).
        """
        self.__target_pos = target_pos.copy() if target_pos is not None else None
