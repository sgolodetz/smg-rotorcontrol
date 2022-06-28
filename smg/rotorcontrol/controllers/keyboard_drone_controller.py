import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from typing import List, Optional, Sequence, Tuple

from smg.rotory.drones import Drone

from .drone_controller import DroneController


class KeyboardDroneController(DroneController):
    """A keyboard-based flight controller for a drone."""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        Construct a keyboard-based flight controller for a drone.

        :param drone:   The drone.
        """
        super().__init__()
        self.__drone: Drone = drone

    # PUBLIC METHODS

    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
                image: np.ndarray, image_timestamp: Optional[float] = None,
                intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        .. note::
            This controller requires the drone's rates to be calibrated. We explicitly check for this.

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
        # If the drone's rates have not been calibrated, raise an exception and early out.
        if not self.__drone.has_calibrated_rates():
            raise RuntimeError("Error: Drones must have calibrated rates when using keyboard control")

        # If no PyGame events were passed in, use an empty list of events as the default.
        if events is None:
            events = []

        # Get the keys that are currently being pressed by the user.
        pressed_keys: Sequence[bool] = pygame.key.get_pressed()

        # Process any PyGame events that have happened since the last iteration.
        for event in events:
            if event.type == pygame.KEYDOWN:
                # If the user presses the 'u' and 'left shift' keys, take off.
                if event.key == pygame.K_u and pressed_keys[pygame.K_LSHIFT]:
                    self.__drone.takeoff()

                # If the user presses the 'o' and 'left shift' keys, land.
                elif event.key == pygame.K_o and pressed_keys[pygame.K_LSHIFT]:
                    self.__drone.land()

        # Allow the user to control the forward/backward movement of the drone.
        if pressed_keys[pygame.K_i]:
            self.__drone.move_forward(self.__drone.calculate_forward_rate(m_per_s=1.0))
        elif pressed_keys[pygame.K_k]:
            self.__drone.move_forward(self.__drone.calculate_forward_rate(m_per_s=-1.0))
        else:
            self.__drone.move_forward(0.0)

        # Allow the user to control the left/right turning movement of the drone.
        if pressed_keys[pygame.K_j] and pressed_keys[pygame.K_LSHIFT]:
            self.__drone.turn(self.__drone.calculate_turn_rate(rad_per_s=np.deg2rad(-60)))
        elif pressed_keys[pygame.K_l] and pressed_keys[pygame.K_LSHIFT]:
            self.__drone.turn(self.__drone.calculate_turn_rate(rad_per_s=np.deg2rad(60)))
        else:
            self.__drone.turn(0.0)

        # Allow the user to control the left/right strafing movement of the drone.
        if pressed_keys[pygame.K_j] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_right(self.__drone.calculate_right_rate(m_per_s=-1.0))
        elif pressed_keys[pygame.K_l] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_right(self.__drone.calculate_right_rate(m_per_s=1.0))
        else:
            self.__drone.move_right(0.0)

        # Allow the user to control the upward/downward movement of the drone.
        if pressed_keys[pygame.K_u] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_up(self.__drone.calculate_up_rate(m_per_s=1.0))
        elif pressed_keys[pygame.K_o] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_up(self.__drone.calculate_up_rate(m_per_s=-1.0))
        else:
            self.__drone.move_up(0.0)
