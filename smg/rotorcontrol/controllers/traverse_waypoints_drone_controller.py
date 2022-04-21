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
    """
    A flight controller for a drone that tries to traverse a specified set of waypoints.

    .. note::
        The waypoints can be changed on-the-fly as desired.
    """

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, planning_octree: OcTree):
        """
        Construct a flight controller for a drone that tries to traverse a specified set of waypoints.

        .. note::
            The planning octree is not thread-safe and *must not* be used by multiple drone controllers at once.

        :param debug:           Whether to enable debugging.
        :param drone:           The drone.
        :param planning_octree: The planning octree (used for path planning).
        """
        self.__alive: bool = False

        self.__ay: float = 10
        self.__debug: bool = debug
        self.__drone: Drone = drone
        self.__planning_octree: OcTree = planning_octree
        self.__should_terminate: threading.Event = threading.Event()

        # The path planning variables, together with their lock.
        self.__current_path: Optional[Path] = None
        self.__current_pos: Optional[np.ndarray] = None
        self.__new_path_available: bool = False
        self.__new_waypoint_count: int = 0
        self.__planning_lock: threading.Lock = threading.Lock()
        self.__planning_path: Optional[Path] = None
        self.__waypoints: List[np.ndarray] = []

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

    def append_waypoints(self, new_waypoints: List[np.ndarray]) -> None:
        """
        Append one or more waypoints to the list of waypoints that the drone should traverse.

        :param new_waypoints:   The new waypoints to append.
        """
        with self.__planning_lock:
            self.__waypoints += new_waypoints
            self.__new_waypoint_count += len(new_waypoints)

    def get_current_path(self) -> Optional[Path]:
        """Get a copy of the current path (if any), or None otherwise."""
        return self.__current_path.copy() if self.__current_path is not None else None

    def get_occupancy_colourer(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get a function that can be used to colour waypoints on a path based on their occupancy status."""
        return self.__planning_toolkit.occupancy_colourer()

    def get_planning_toolkit(self) -> PlanningToolkit:
        """TODO"""
        return self.__planning_toolkit

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
        # If no tracker pose has been passed in, raise an exception and early out.
        if tracker_c_t_i is None:
            raise RuntimeError("Tracker poses must be provided when using 'traverse waypoints' control")

        # Extract the current position of the drone from the tracker pose provided.
        tracker_i_t_c: np.ndarray = np.linalg.inv(tracker_c_t_i)
        current_pos: np.ndarray = tracker_i_t_c[0:3, 3]

        # If we're able to acquire the planning lock:
        acquired: bool = self.__planning_lock.acquire(blocking=False)
        if acquired:
            try:
                # If there's a new path available from the path planner, update the current path and reset the flag.
                if self.__new_path_available:
                    self.__current_path = self.__planning_path.copy() if self.__planning_path is not None else None
                    self.__new_path_available = False

                # Otherwise, update the path planner's view of the current path.
                else:
                    self.__planning_path = self.__current_path.copy() if self.__current_path is not None else None

                # Provide the path planner with the current position of the drone.
                self.__current_pos = current_pos.copy()

                # If there are any waypoints through which a path has not yet been planned, tell the path planner
                # that some path planning is needed.
                if self.__new_waypoint_count > 0:
                    self.__planning_is_needed = True
                    self.__planning_needed.notify()

                # Check whether the drone has reached the first waypoint (if any), and remove it from the list if so.
                # FIXME: Stop hard-coding the threshold here.
                if len(self.__waypoints) > 0 and np.linalg.norm(current_pos - self.__waypoints[0]) < 0.025:
                    self.__waypoints = self.__waypoints[1:]
            finally:
                # Make sure that the planning lock is released before carrying on.
                self.__planning_lock.release()

        # If there's a current path, update it based on the drone's current position.
        if self.__current_path is not None:
            if self.__debug:
                start = timer()

            new_path = self.__planner.update_path(
                current_pos, self.__current_path, debug=self.__debug,
                d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                allow_shortcuts=True, pull_strings=True, use_clearance=True,
                nearest_waypoint_tolerance=0.025
            )

            if new_path is None:
                new_path = self.__planner.update_path(
                    current_pos, self.__current_path, debug=self.__debug,
                    d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                    allow_shortcuts=True, pull_strings=True, use_clearance=False,
                    nearest_waypoint_tolerance=0.025
                )

            self.__current_path = new_path

            if self.__debug:
                end = timer()
                # noinspection PyUnboundLocalVariable
                print(f"Path Updating: {end - start}s")

        # A flag indicating whether or not the drone should stop moving.
        stop_drone: bool = True

        # If there's still a current path, try to follow it.
        if self.__current_path is not None:
            cam: SimpleCamera = CameraPoseConverter.pose_to_camera(tracker_c_t_i)
            offset: np.ndarray = self.__current_path[1].position - current_pos
            offset_length: float = np.linalg.norm(offset)
            if offset_length >= 1e-4:
                stop_drone = False

                current_n: np.ndarray = vg.normalize(np.array([cam.n()[0], 0, cam.n()[2]]))
                target_n: np.ndarray = vg.normalize(np.array([offset[0], 0, offset[2]]))
                cp: np.ndarray = np.cross(current_n, target_n)
                sign: int = 1 if np.dot(cp, np.array([0, -1, 0])) >= 0 else -1
                angle: float = 0.0
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        angle = sign * np.arccos(np.clip(np.dot(current_n, target_n), -1.0, 1.0))
                    except RuntimeWarning:
                        print(current_n, target_n, np.dot(current_n, target_n))
                turn_rate: float = np.clip(-angle / (np.pi / 2), -1.0, 1.0)
                normalized_offset: np.ndarray = offset / offset_length
                speed: float = 0.5
                forward_rate: float = vg.scalar_projection(normalized_offset, cam.n()) * speed
                right_rate: float = vg.scalar_projection(normalized_offset, -cam.u()) * speed
                up_rate: float = vg.scalar_projection(normalized_offset, cam.v()) * speed

                # if pygame.key.get_pressed()[pygame.K_c]:
                if True:
                    self.__drone.turn(turn_rate)
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

        # If the drone should stop moving, stop it.
        if stop_drone:
            self.__drone.stop()

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        """
        Set the waypoints that the drone should traverse.

        :param waypoints:   The waypoints that the drone should traverse.
        """
        with self.__planning_lock:
            self.__waypoints = waypoints.copy()
            self.__new_waypoint_count = len(waypoints)

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
            # can use them without having to hold on to the lock.
            with self.__planning_lock:
                while not self.__planning_is_needed:
                    self.__planning_needed.wait(0.1)
                    if self.__should_terminate.is_set():
                        return

                current_pos: Optional[np.ndarray] = self.__current_pos.copy()
                new_waypoint_count: int = self.__new_waypoint_count
                waypoints: List[np.ndarray] = self.__waypoints.copy()

            # Determine whether we're appending to the path or replacing it, and set the waypoints for which
            # to perform path planning accordingly.
            appending_waypoints: bool = new_waypoint_count != len(waypoints)
            if appending_waypoints:
                waypoints_to_plan: List[np.ndarray] = waypoints[-new_waypoint_count - 1:]
            else:
                waypoints_to_plan: List[np.ndarray] = [current_pos] + waypoints

            # Plan a path through the chosen waypoints.
            if self.__debug:
                start = timer()

            new_path: Optional[Path] = self.__planner.plan_multi_step_path(
                waypoints_to_plan,
                d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                allow_shortcuts=True, pull_strings=True, use_clearance=True
            )

            if new_path is None:
                new_path = self.__planner.plan_multi_step_path(
                    waypoints_to_plan,
                    d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                    allow_shortcuts=True, pull_strings=True, use_clearance=False
                )

            if self.__debug:
                end = timer()
                # noinspection PyUnboundLocalVariable
                print(f"Path Planning: {end - start}s")

            # Update the shared path variables so that the new path can be picked up by other threads.
            with self.__planning_lock:
                # If we're appending waypoints to the existing path, append the planned sub-path to what's left of
                # the current path.
                if appending_waypoints:
                    if new_path is not None:
                        if self.__planning_path is not None:
                            self.__planning_path = new_path.replace_before(0, self.__planning_path, keep_last=False)
                        else:
                            self.__planning_path = new_path
                            self.__planning_path.positions[0] = current_pos

                # Otherwise, simply replace the existing path.
                else:
                    self.__planning_path = new_path

                self.__new_path_available = True
                self.__new_waypoint_count -= new_waypoint_count
                self.__planning_is_needed = False

            # Wait for 10ms before performing any further path planning, so as to avoid a spin loop.
            time.sleep(0.01)
