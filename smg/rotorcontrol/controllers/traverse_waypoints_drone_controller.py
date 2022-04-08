import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# noinspection PyPackageRequirements
import pygame
import threading
import time
import vg

from timeit import default_timer as timer
from typing import Callable, List, Optional, Tuple

from smg.navigation import AStarPathPlanner, OCS_OCCUPIED, Path, PlanningToolkit
from smg.pyoctomap import OcTree
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController


class TraverseWaypointsDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, planning_octree: OcTree):
        self.__alive: bool = False

        self.__debug: bool = debug
        self.__drone: Drone = drone
        self.__planning_octree: OcTree = planning_octree
        self.__should_terminate: threading.Event = threading.Event()

        # The path planning variables, together with their lock.
        self.__current_pos: Optional[np.ndarray] = None
        self.__interpolated_path: Optional[Path] = None
        self.__path: Optional[Path] = None
        self.__planning_lock: threading.Lock = threading.Lock()
        self.__waypoints: List[np.ndarray] = []
        self.__waypoints_changed: bool = False

        # Construct the planning toolkit.
        self.__planning_toolkit = PlanningToolkit(
            self.__planning_octree,
            neighbours=PlanningToolkit.neighbours6,
            node_is_free=lambda n: self.__planning_toolkit.occupancy_status(n) != OCS_OCCUPIED
        )

        # Construct the path planner.
        self.__planner: AStarPathPlanner = AStarPathPlanner(self.__planning_toolkit, debug=False)

        # Set up the path planning thread and its associated variables.
        self.__planning_is_needed: bool = False
        self.__planning_needed: threading.Condition = threading.Condition(self.__planning_lock)
        self.__planning_thread: threading.Thread = threading.Thread(target=self.__run_planning)

        # Start the path planning thread.
        self.__planning_thread.start()

        self.__alive = True

    # PUBLIC METHODS

    def get_interpolated_path(self) -> Optional[Path]:
        """TODO"""
        with self.__planning_lock:
            return self.__interpolated_path.copy() if self.__interpolated_path is not None else None

    def get_occupancy_colourer(self) -> Callable[[np.ndarray], np.ndarray]:
        """TODO"""
        return self.__planning_toolkit.occupancy_colourer()

    def get_path(self) -> Optional[Path]:
        """TODO"""
        with self.__planning_lock:
            return self.__path.copy() if self.__path is not None else None

    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
                image: np.ndarray, image_timestamp: Optional[float] = None,
                intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        .. note::
            This controller (i) requires the tracker poses to be passed in, and (ii) requires that they be metric.
            We explicitly check (i). We can't check (ii), so client code is responsible for correct use.

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
        if tracker_c_t_i is None:
            raise RuntimeError("Tracker poses must be provided when using 'traverse waypoints' control")

        # noinspection PyUnusedLocal
        path: Optional[Path] = None

        # If we're able to acquire the planning lock:
        acquired: bool = self.__planning_lock.acquire(blocking=False)
        if acquired:
            try:
                # Extract the current position of the drone from the tracker pose provided.
                tracker_i_t_c: np.ndarray = np.linalg.inv(tracker_c_t_i)
                current_pos: np.ndarray = tracker_i_t_c[0:3, 3]

                # Check whether the drone has reached the first waypoint (if any), and remove it from the list if so.
                if len(self.__waypoints) > 0 and np.linalg.norm(current_pos - self.__waypoints[0]) < 0.025:
                    self.__waypoints = self.__waypoints[1:]

                # If there are any waypoints remaining, provide the path planner with the current position of the
                # drone, and tell it that some path planning is needed.
                if len(self.__waypoints) > 0:
                    self.__current_pos = current_pos
                    self.__planning_is_needed = True
                    self.__planning_needed.notify()
                else:
                    # TODO: Possible bug fix - check if needed.
                    self.__path = None

                # Make a thread-local copy of any existing path that has been planned so that we can use it
                # without having to hold on to the lock.
                path = self.__path.copy() if self.__path is not None else None
            finally:
                self.__planning_lock.release()

        # Otherwise, early out.
        else:
            return

        # TODO
        if path is not None:
            cam: SimpleCamera = CameraPoseConverter.pose_to_camera(tracker_c_t_i)
            offset: np.ndarray = path[1].position - self.__current_pos
            current_n: np.ndarray = vg.normalize(np.array([cam.n()[0], 0, cam.n()[2]]))
            target_n: np.ndarray = vg.normalize(np.array([offset[0], 0, offset[2]]))
            cp: np.ndarray = np.cross(current_n, target_n)
            sign: int = 1 if np.dot(cp, np.array([0, -1, 0])) >= 0 else -1
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    angle: float = sign * np.arccos(np.clip(np.dot(current_n, target_n), -1.0, 1.0))
                except RuntimeWarning:
                    print(current_n, target_n, np.dot(current_n, target_n))
            rate: float = np.clip(-angle / (np.pi / 2), -1.0, 1.0)
            # print(current_n, target_n, cp, angle, angle * 180 / np.pi, rate)
            angle_to_vertical: float = np.arccos(np.clip(np.dot(vg.normalize(offset), np.array([0, -1, 0])), -1.0, 1.0)) * 180 / np.pi
            normalized_offset: np.ndarray = vg.normalize(offset)
            # print(offset, normalized_offset, vg.scalar_projection(normalized_offset, cam.n()), vg.scalar_projection(normalized_offset, -cam.u()), vg.scalar_projection(normalized_offset, cam.v()))
            speed: float = 0.5
            forward_rate: float = vg.scalar_projection(normalized_offset, cam.n()) * speed
            right_rate: float = vg.scalar_projection(normalized_offset, -cam.u()) * speed
            up_rate: float = vg.scalar_projection(normalized_offset, cam.v()) * speed

            if True:  # pygame.key.get_pressed()[pygame.K_c]:
                self.__drone.turn(rate)
            else:
                self.__drone.turn(0.0)

            # if pygame.key.get_pressed()[pygame.K_c] and angle * 180 / np.pi <= 90.0:
            if angle * 180 / np.pi <= 90.0:
                self.__drone.move_forward(forward_rate)
                self.__drone.move_right(right_rate)
                self.__drone.move_up(up_rate)
            else:
                self.__drone.move_forward(0.0)
                self.__drone.move_right(0.0)
                self.__drone.move_up(0.0)
        else:
            self.__drone.stop()

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        """
        TODO

        :param waypoints:   TODO
        """
        with self.__planning_lock:
            self.__waypoints = waypoints
            self.__waypoints_changed = True

    def terminate(self) -> None:
        """Destroy the controller."""
        if self.__alive:
            # Set the termination flag if it isn't set already.
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()

            # Join any running threads.
            self.__planning_thread.join()

            self.__alive = False

    # PRIVATE METHODS

    def __run_planning(self) -> None:
        """Run the path planning thread."""
        # Until the drone controller should terminate:
        while not self.__should_terminate.is_set():
            # Wait until path planning is required, and then capture the shared variables locally so that we
            # can use them without having to hold on to the lock. Note that we copy the current position and
            # waypoints, as they may be modified during path planning, but we don't need to copy the paths,
            # since the only thread that ever writes to them is this one.
            with self.__planning_lock:
                while not self.__planning_is_needed:
                    self.__planning_needed.wait(0.1)
                    if self.__should_terminate.is_set():
                        return

                current_pos: Optional[np.ndarray] = self.__current_pos.copy()
                interpolated_path: Optional[Path] = self.__interpolated_path
                path: Optional[Path] = self.__path
                waypoints: List[np.ndarray] = self.__waypoints.copy()
                waypoints_changed: bool = self.__waypoints_changed

            # If no path has yet been planned through the waypoints, plan one now. Otherwise, if a path already
            # exists, update it based on the agent's current position.
            ay: float = 10
            if path is None or waypoints_changed:
                if self.__debug:
                    start = timer()

                path = self.__planner.plan_multi_step_path(
                    [current_pos] + waypoints,
                    d=PlanningToolkit.l1_distance(ay=ay), h=PlanningToolkit.l1_distance(ay=ay),
                    allow_shortcuts=True, pull_strings=True, use_clearance=True
                )

                if path is None:
                    path = self.__planner.plan_multi_step_path(
                        [current_pos] + waypoints,
                        d=PlanningToolkit.l1_distance(ay=ay), h=PlanningToolkit.l1_distance(ay=ay),
                        allow_shortcuts=True, pull_strings=True, use_clearance=False
                    )

                if self.__debug:
                    end = timer()
                    # noinspection PyUnboundLocalVariable
                    print(f"Path Planning: {end - start}s")
            elif len(path) > 1:
                if self.__debug:
                    start = timer()

                new_path: Optional[Path] = self.__planner.update_path(
                    current_pos, path, debug=self.__debug,
                    d=PlanningToolkit.l1_distance(ay=ay), h=PlanningToolkit.l1_distance(ay=ay),
                    allow_shortcuts=True, pull_strings=True, use_clearance=True,
                    nearest_waypoint_tolerance=0.05
                )

                if new_path is None:
                    new_path = self.__planner.update_path(
                        current_pos, path, debug=self.__debug,
                        d=PlanningToolkit.l1_distance(ay=ay), h=PlanningToolkit.l1_distance(ay=ay),
                        allow_shortcuts=True, pull_strings=True, use_clearance=False,
                        nearest_waypoint_tolerance=0.05
                    )

                path = new_path

                if self.__debug:
                    end = timer()
                    print(f"Path Updating: {end - start}s")

            # Perform curve fitting and interpolation on any path found.
            if path is not None:
                if self.__debug:
                    start = timer()

                interpolated_path = path.interpolate()

                if self.__debug:
                    end = timer()
                    print(f"Path Interpolation: {end - start}s")

            # Update the shared path variables so that the new path and its interpolated variant can be picked up
            # by other threads.
            with self.__planning_lock:
                self.__interpolated_path = interpolated_path
                self.__path = path
                self.__planning_is_needed = False

            # Wait for 10ms before performing any further path planning, so as to avoid a spin loop.
            time.sleep(0.01)
