import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import threading
import time

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

from smg.navigation import AStarPathPlanner, Path, PlanningToolkit
from smg.opengl import OpenGLUtil
from smg.rotory.drones import Drone

from .drone_controller import DroneController
from .traverse_path_drone_controller import TraversePathDroneController


class TraverseWaypointsDroneController(DroneController):
    """
    A flight controller for a drone that tries to traverse a specified set of waypoints.

    .. note::
        The waypoints can be changed on-the-fly as desired.
    """

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, planning_toolkit: PlanningToolkit):
        """
        Construct a flight controller for a drone that tries to traverse a specified set of waypoints.

        :param debug:               Whether to enable debugging.
        :param drone:               The drone.
        :param planning_toolkit:    The planning toolkit (used for path planning).
        """
        super().__init__()

        self.__alive: bool = False

        self.__ay: float = 10
        self.__debug: bool = debug
        self.__drone: Drone = drone
        self.__path_tracking_range: float = 0.05
        self.__planning_toolkit: PlanningToolkit = planning_toolkit
        self.__should_terminate: threading.Event = threading.Event()
        self.__traverse_path_controller: TraversePathDroneController = TraversePathDroneController(drone=drone)
        self.__waypoint_capture_range: float = 0.025

        # The shared variables, together with their lock.
        self.__current_pos: Optional[np.ndarray] = None
        self.__lock: threading.Lock = threading.Lock()
        self.__new_waypoint_count: int = 0
        self.__path: Optional[Path] = None
        self.__stop_planning: Optional[threading.Event] = None
        self.__waypoints: List[np.ndarray] = []

        # Construct the path planner.
        self.__planner: AStarPathPlanner = AStarPathPlanner(self.__planning_toolkit, debug=False)

        # Set up the path planning thread and its associated variables.
        self.__planning_is_needed: bool = False
        self.__planning_needed: threading.Condition = threading.Condition(self.__lock)
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
        with self.__lock:
            self.__waypoints += new_waypoints
            self.__new_waypoint_count += len(new_waypoints)

    def get_current_pos(self) -> Optional[np.ndarray]:
        """Get the current position of the drone (if available yet), or None otherwise."""
        with self.__lock:
            return self.__current_pos.copy() if self.__current_pos is not None else None

    def get_expected_end_pos(self) -> Optional[np.ndarray]:
        """
        Get the expected position of the drone once the controller has finished (if known).

        :return:    The expected position of the drone once the controller has finished, if known, or None otherwise.
        """
        waypoints: List[np.ndarray] = self.get_waypoints()
        return waypoints[-1] if len(waypoints) > 0 else None

    def get_expected_end_state(self) -> Optional[Drone.EState]:
        """
        Get the expected state of the drone once the controller has finished (if known).

        :return:    The expected state of the drone once the controller has finished, if known, or None otherwise.
        """
        return Drone.FLYING

    def get_new_waypoints(self) -> List[np.ndarray]:
        """Get any new (i.e. as yet unplanned) waypoints that the drone should traverse."""
        with self.__lock:
            return self.__waypoints[-self.__new_waypoint_count:] if self.__new_waypoint_count > 0 else []

    def get_path(self) -> Optional[Path]:
        """Get a copy of the path (if any), or None otherwise."""
        return self.__path.copy() if self.__path is not None else None

    def get_waypoints(self) -> List[np.ndarray]:
        """Get the waypoints that the drone should traverse."""
        with self.__lock:
            return self.__waypoints.copy()

    def has_finished(self) -> bool:
        """
        Get whether or not the controller has finished.

        :return:    True, if the controller has finished, or False otherwise.
        """
        with self.__lock:
            return len(self.__waypoints) == 0

    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
                image: np.ndarray, image_timestamp: Optional[float] = None,
                intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        .. note::
            This controller (i) requires the tracker poses to be passed in, and (ii) requires that they be
            scale-correct. We explicitly check (i). We can't check (ii), so client code is responsible for
            correct use.
        .. note::
            This controller also requires the drone's rates to be calibrated. We explicitly check for this.

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
        # forwarded on to the 'traverse path' controller.
        kwargs: Dict[str, Any] = {key: value for key, value in locals().items() if key != "self"}

        # If no tracker pose has been passed in, raise an exception and early out.
        if tracker_c_t_i is None:
            raise RuntimeError("Error: Tracker poses must be provided when using 'traverse waypoints' control")

        # If the drone's rates have not been calibrated, raise an exception and early out.
        if not self.__drone.has_calibrated_rates():
            raise RuntimeError("Error: Drones must have calibrated rates when using 'traverse waypoints' control")

        with self.__lock:
            # --- Step 1: Update the drone's current position, and ensure its estimated start position is set ---#

            # Extract the current position of the drone from the tracker pose provided.
            self.__current_pos = DroneController._extract_current_pos(tracker_c_t_i)

            # Set the estimated start position to the current position of the drone if it's not already known.
            if self.get_expected_start_pos() is None:
                self.set_expected_start_pos(self.__current_pos.copy())

            # --- Step 2: Trigger Path Planning (if required) ---#

            # If there are any waypoints through which a path has not yet been planned, tell the path planner
            # that some path planning is needed.
            if self.__new_waypoint_count > 0:
                self.__planning_is_needed = True
                self.__planning_needed.notify()

            # --- Step 3: Update Current Path --- #

            # If there's a current path, update it based on the drone's current position.
            if self.__path is not None:
                if self.__debug:
                    start = timer()

                # First try to update the path whilst maintaining sufficient clearance.
                new_path = self.__planner.update_path(
                    self.__current_pos, self.__path, debug=self.__debug,
                    d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                    allow_shortcuts=True, pull_strings=True, use_clearance=True,
                    path_tracking_range=self.__path_tracking_range,
                    waypoint_capture_range=self.__waypoint_capture_range
                )

                # If that doesn't work, fall back to updating the path without requiring sufficient clearance,
                # on the basis that it will probably work in any case.
                if new_path is None:
                    new_path = self.__planner.update_path(
                        self.__current_pos, self.__path, debug=self.__debug,
                        d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                        allow_shortcuts=True, pull_strings=True, use_clearance=False,
                        path_tracking_range=self.__path_tracking_range,
                        waypoint_capture_range=self.__waypoint_capture_range
                    )

                self.__path = new_path

                # Check whether the drone has reached the first essential waypoint (if any), and remove it if so.
                next_waypoint_distance: float = np.linalg.norm(self.__current_pos - self.__waypoints[0])
                if len(self.__waypoints) > 0 and next_waypoint_distance <= self.__waypoint_capture_range:
                    self.__waypoints = self.__waypoints[1:]

                if self.__debug:
                    end = timer()
                    # noinspection PyUnboundLocalVariable
                    print(f"Path Updating: {end - start}s")

            # --- Step 4: Make Drone Follow Current Path --- #

            # Delegate path following to the 'traverse path' controller.
            self.__traverse_path_controller.set_path(self.__path)
            self.__traverse_path_controller.iterate(**kwargs)

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # Render the path that the drone is following (if any).
        path: Optional[Path] = self.get_path()
        if path is not None:
            path.render(
                start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                waypoint_colourer=self.__planning_toolkit.occupancy_colourer()
            )

        # Render any new waypoints for which a path has not yet been planned.
        # FIXME: This is currently a bit messy - it needs tidying up and moving somewhere more sensible.
        glColor3f(1, 1, 0)

        new_waypoints: List[np.ndarray] = self.get_new_waypoints()
        waypoints: List[np.ndarray] = self.get_waypoints()
        last_waypoint: Optional[np.ndarray] = self.get_current_pos()
        if last_waypoint is None:
            last_waypoint = self.get_expected_start_pos()
        if path is not None and len(new_waypoints) != len(waypoints):
            last_waypoint = path[-1].position

        if last_waypoint is not None:
            glLineWidth(5)
            for i in range(len(new_waypoints)):
                OpenGLUtil.render_sphere(new_waypoints[i], 0.1, slices=10, stacks=10)
                glBegin(GL_LINES)
                glVertex3f(*last_waypoint)
                glVertex3f(*new_waypoints[i])
                glEnd()
                last_waypoint = new_waypoints[i]
            glLineWidth(1)

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        """
        Set the waypoints that the drone should traverse.

        :param waypoints:   The waypoints that the drone should traverse.
        """
        with self.__lock:
            # Delete the current path (if any).
            self.__path = None

            # Stop any ongoing path planning.
            if self.__stop_planning is not None:
                self.__stop_planning.set()

            # Replace the waypoints, and update the number of new waypoints for which path planning will be needed.
            self.__waypoints = waypoints.copy()
            self.__new_waypoint_count = len(waypoints)

    def terminate(self) -> None:
        """Destroy the controller."""
        if self.__alive:
            # Set the termination flag if it isn't set already.
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()

            # Stop any ongoing path planning.
            with self.__lock:
                if self.__stop_planning is not None:
                    self.__stop_planning.set()

            # Join any running threads.
            self.__planning_thread.join()

            self.__alive = False

    # PRIVATE METHODS

    def __run_planning(self) -> None:
        """Run the path planning thread."""
        # Until the drone controller should terminate:
        while not self.__should_terminate.is_set():
            with self.__lock:
                # Wait until path planning is required.
                while not self.__planning_is_needed:
                    self.__planning_needed.wait(0.1)
                    if self.__should_terminate.is_set():
                        return

                # Capture the shared variables locally so that we can use them without having to hold on to the lock.
                current_pos: Optional[np.ndarray] = self.__current_pos.copy()
                new_waypoint_count: int = self.__new_waypoint_count
                waypoints: List[np.ndarray] = self.__waypoints.copy()

                # Construct a new threading event that can be used to stop path planning if necessary.
                stop_planning: threading.Event = threading.Event()
                self.__stop_planning = stop_planning

            # Determine whether we're appending to the path or replacing it, and set the waypoints for which
            # to perform path planning accordingly.
            appending_waypoints: bool = new_waypoint_count != len(waypoints)
            if appending_waypoints:
                waypoints_to_plan: List[np.ndarray] = waypoints[-new_waypoint_count - 1:]
            else:
                waypoints_to_plan: List[np.ndarray] = [current_pos] + waypoints

            # Plan a path through the chosen waypoints. First try to plan a path that maintains sufficient clearance.
            # If that doesn't work, try to plan a path without requiring sufficient clearance, on the basis that it
            # will probably work in any case.
            if self.__debug:
                start = timer()

            new_path: Optional[Path] = self.__planner.plan_multi_step_path(
                waypoints_to_plan,
                d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                allow_shortcuts=True, pull_strings=True, use_clearance=True, stop_planning=stop_planning
            )

            if new_path is None:
                new_path = self.__planner.plan_multi_step_path(
                    waypoints_to_plan,
                    d=PlanningToolkit.l1_distance(ay=self.__ay), h=PlanningToolkit.l1_distance(ay=self.__ay),
                    allow_shortcuts=True, pull_strings=True, use_clearance=False, stop_planning=stop_planning
                )

            if self.__debug:
                end = timer()
                # noinspection PyUnboundLocalVariable
                print(f"Path Planning: {end - start}s")

            with self.__lock:
                # Provided path planning wasn't stopped, update the shared variables so that the new path can be
                # picked up by other threads.
                if not self.__stop_planning.is_set():
                    # If we're appending waypoints to the existing path, append the planned sub-path to what's left
                    # of the current path.
                    if appending_waypoints:
                        # If we successfully planned a sub-path to append to the existing path:
                        if new_path is not None:
                            # If the existing path still exists (bearing in mind that the drone's been flying along
                            # it whilst we've been planning the sub-path to append):
                            if self.__path is not None:
                                # Prepend what's left of the existing path to the planned sub-path.
                                self.__path = new_path.replace_before(0, self.__path, keep_last=False)

                            # Otherwise:
                            else:
                                # Replace the existing path, updating its starting waypoint to match the current
                                # position of the drone in the process (note that this may not be exactly the same
                                # as the goal of the original path, since the drone may not have perfectly made it
                                # to the goal).
                                self.__path = new_path
                                self.__path.positions[0] = self.__current_pos.copy()

                    # Otherwise, simply replace the existing path.
                    else:
                        self.__path = new_path

                    # Decrease the number of new waypoints for which planned is still required, and reset the flag
                    # indicating that planning is needed. If there are still waypoints outstanding, path planning
                    # will shortly be triggered again by the main thread in any case.
                    self.__new_waypoint_count -= new_waypoint_count
                    self.__planning_is_needed = False

                # Delete the event that was provided to allow path planning to be stopped - it's no longer usable.
                self.__stop_planning = None

            # Wait for 10ms before performing any further path planning, so as to avoid a spin loop.
            time.sleep(0.01)
