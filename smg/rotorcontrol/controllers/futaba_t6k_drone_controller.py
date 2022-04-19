import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame

from typing import List, Optional, Tuple

from smg.joysticks import FutabaT6K

from .drone_controller import DroneController
from smg.rotory.drones import Drone


class FutabaT6KDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        TODO

        :param drone:   TODO
        """
        self.__can_move_gimbal: bool = False
        self.__drone: Drone = drone

        # Try to determine the joystick index of the Futaba T6K. If no joystick is plugged in, early out.
        joystick_count: int = pygame.joystick.get_count()
        joystick_idx: int = 0
        if joystick_count == 0:
            exit(0)
        elif joystick_count != 1:
            # TODO: Prompt the user for the joystick to use.
            pass

        # Construct and calibrate the Futaba T6K.
        self.__joystick: FutabaT6K = FutabaT6K(joystick_idx)
        self.__joystick.calibrate()

    # PUBLIC METHODS

    def has_finished(self) -> bool:
        """
        Get whether or not the controller has finished.

        :return:    True, if the controller has finished, or False otherwise.
        """
        # The controller has finished if both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
        return self.__joystick.get_button(0) == 0 and self.__joystick.get_button(1) == 0

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
        # TODO
        if events is None:
            events = []

        # TODO
        for event in events:
            if event.type == pygame.JOYBUTTONDOWN:
                # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                if event.button == 0:
                    self.__drone.takeoff()
            elif event.type == pygame.JOYBUTTONUP:
                # If Button 0 on the Futaba T6K is set to its "released" state, land.
                if event.button == 0:
                    self.__drone.land()

        # Update the movement of the drone based on the pitch, roll and yaw values output by the Futaba T6K.
        self.__drone.move_forward(self.__joystick.get_pitch())
        self.__drone.turn(self.__joystick.get_yaw())

        if self.__joystick.get_button(1) == 0:
            self.__drone.move_right(0)
            self.__drone.move_up(self.__joystick.get_roll())
        else:
            self.__drone.move_right(self.__joystick.get_roll())
            self.__drone.move_up(0)

        # If the throttle goes above half-way, enable movement of the drone's gimbal from now on.
        throttle: float = self.__joystick.get_throttle()
        if throttle >= 0.5:
            self.__can_move_gimbal = True

        # If the drone's gimbal can be moved, update its pitch based on the current value of the throttle.
        # Note that the throttle value is in [0,1], so we rescale it to a value in [-1,1] as a first step.
        if self.__can_move_gimbal:
            self.__drone.update_gimbal_pitch(2 * (self.__joystick.get_throttle() - 0.5))
