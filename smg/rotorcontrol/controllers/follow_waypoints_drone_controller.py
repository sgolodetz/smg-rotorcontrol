import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame
import threading
import time

from typing import List, Optional, Tuple

from smg.navigation import AStarPathPlanner, OCS_OCCUPIED, Path, PlanningToolkit
from smg.pyoctomap import OcTree
from smg.rotory.drones import Drone

from .drone_controller import DroneController


class FollowWaypointsDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone, planning_octree: OcTree):
        self.__alive: bool = False

        self.__drone: Drone = drone
        self.__planning_octree: OcTree = planning_octree
        self.__should_terminate: threading.Event = threading.Event()

        # The path planning variables, together with their lock.
        self.__path: Optional[Path] = None
        self.__planning_lock: threading.Lock = threading.Lock()
        self.__waypoints: Optional[List[np.ndarray]] = None

        # Construct the planning toolkit.
        self.__planning_toolkit = PlanningToolkit(
            self.__planning_octree,
            neighbours=PlanningToolkit.neighbours6,
            node_is_free=lambda n: self.__planning_toolkit.occupancy_status(n) != OCS_OCCUPIED
        )

        # Construct the path planner.
        self.__planner: AStarPathPlanner = AStarPathPlanner(self.__planning_toolkit, debug=False)

        # Start the path planning thread.
        self.__planning_thread: threading.Thread = threading.Thread(target=self.__run_planning)
        self.__planning_thread.start()

        self.__alive = True

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
        :param tracker_c_t_i:       A transformation from initial camera space to current camera space, as estimated
                                    by any tracker that's running (optional). Note that if the tracker is a monocular
                                    one, the transformation will be non-metric.
        """
        pass

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        """
        TODO

        :param waypoints:   TODO
        """
        with self.__planning_lock:
            self.__waypoints = waypoints

    def terminate(self) -> None:
        """Destroy the controller."""
        if self.__alive:
            # Set the termination flag if it isn't set already.
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()

            # Join any running threads.
            # TODO

            self.__alive = False

    # PRIVATE METHODS

    def __run_planning(self) -> None:
        # Until the simulator should terminate:
        while not self.__should_terminate.is_set():
            # Wait for 10ms before performing any further path planning, so as to avoid a spin loop.
            time.sleep(0.01)
