import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import vg

from collections import deque
from OpenGL.GL import *
from typing import Any, cast, Deque, Dict, Optional, List, Tuple

from smg.navigation import PathNode, PlanningToolkit
from smg.opengl import OpenGLUtil
from smg.pyoctomap import OctomapPicker
from smg.rigging.cameras import Camera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController
from .landing_drone_controller import LandingDroneController
from .takeoff_drone_controller import TakeoffDroneController
from .traverse_waypoints_drone_controller import TraverseWaypointsDroneController


class RTSStyleDroneController(DroneController):
    """An RTS("real-time strategy")-style flight controller for a drone."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, picker: Optional[OctomapPicker],
                 planning_toolkit: Optional[PlanningToolkit], viewing_camera: Camera):
        """
        Construct an RTS-style flight controller for a drone.

        :param debug:               Whether to enable debugging.
        :param drone:               The drone.
        :param picker:              A picker for the scene (required: an exception will be raised if this is None).
        :param planning_toolkit:    The planning toolkit (used for path planning).
        :param viewing_camera:      The virtual camera being used to view the scene.
        """
        super().__init__()

        if picker is None:
            raise RuntimeError("Error: An RTS-style drone controller requires a picker for the scene")
        if planning_toolkit is None:
            raise RuntimeError("Error: An RTS-style drone controller requires a planning toolkit for the scene")

        self.__debug: bool = debug
        self.__drone: Drone = drone
        self.__goal_pos: Optional[np.ndarray] = None
        self.__ground_pos: Optional[np.ndarray] = None
        self.__height_offset: float = 1.0
        self.__inner_controllers: Deque[DroneController] = deque()
        self.__left_mouse_down: bool = False
        self.__movement_allowed: bool = True
        self.__orienting_pos: Optional[np.ndarray] = None
        self.__picker: OctomapPicker = cast(OctomapPicker, picker)
        self.__planning_toolkit: PlanningToolkit = cast(PlanningToolkit, planning_toolkit)
        self.__pre_goal_pos: Optional[np.ndarray] = None
        self.__viewing_camera: Camera = viewing_camera

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
        # forwarded on to the inner controller.
        kwargs: Dict[str, Any] = {key: value for key, value in locals().items() if key != "self"}

        # If no PyGame events were passed in, use an empty list of events as the default.
        if events is None:
            events = []

        # Update the goal that the user wants the drone to achieve.
        self.__update_goal()

        # Process any PyGame events that have happened since the last iteration.
        for event in events:
            # If the user presses the 'space' key, toggle whether the drone is allowed to move.
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.__movement_allowed = not self.__movement_allowed

            # If the user presses any other key, or clicks or releases a mouse button:
            elif event.type == pygame.KEYDOWN \
                    or event.type == pygame.MOUSEBUTTONDOWN \
                    or event.type == pygame.MOUSEBUTTONUP:
                # If the user is currently pressing one of the shift keys (indicating that an append is desired),
                # and the last inner controller is not None (indicating that an append is possible):
                last_inner_controller: Optional[DroneController] = self.__get_last_inner_controller()

                if (pygame.key.get_mods() & pygame.KMOD_SHIFT) and last_inner_controller is not None:
                    # Try to make and append a new inner controller.
                    self.__try_append_new_inner_controller(event, last_inner_controller)

                # Otherwise:
                else:
                    # Extract the current position of the drone from the tracker pose provided.
                    drone_pos: np.ndarray = DroneController._extract_current_pos(tracker_c_t_i)

                    # Try to make and set a new inner controller.
                    self.__try_set_new_inner_controller(event, drone_pos)

            # If the user scrolls the mouse wheel, change the desired offset of the goal position above the floor.
            elif event.type == pygame.MOUSEWHEEL:
                self.__height_offset = np.clip(self.__height_offset + event.y * 0.2, 0.3, 3.0)

        # If there is an active inner controller and the drone is currently allowed to move, delegate lower-level
        # control of the drone to the active inner controller; otherwise, stop the drone.
        active_inner_controller: Optional[DroneController] = self.__get_active_inner_controller()
        if active_inner_controller is not None and self.__movement_allowed:
            active_inner_controller.iterate(**kwargs)
        else:
            self.__drone.stop()

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # Render the user interfaces for the inner controllers.
        for inner_controller in self.__inner_controllers:
            inner_controller.render_ui()

        # If a goal position has been determined:
        if self.__goal_pos is not None:
            # Render a sphere at the goal position with a colour that indicates the traversability of the goal node.
            self.__render_traversability_sphere(self.__goal_pos, radius=0.1)

            # Render a vertical cylinder that joins the goal position to the closest point on the ground beneath it.
            # This makes it easier for the user to see where the goal lies in relation to the scene.
            glColor3f(1, 0, 0)
            OpenGLUtil.render_cylinder(self.__goal_pos, self.__ground_pos, 0.025, 0.025, slices=10)

            # If a pre-goal position (used when specifying the goal orientation) has been determined:
            if self.__pre_goal_pos is not None:
                # Render a sphere at the pre-goal position with a colour that indicates the traversability of the
                # pre-goal node.
                self.__render_traversability_sphere(self.__pre_goal_pos, radius=0.1)

                # Render an arrow to show the goal orientation, using a colour that indicates its validity.
                if self.__goal_orientation_valid:
                    glColor3f(0, 1, 0)
                else:
                    glColor3f(1, 0, 0)

                join_pos: np.ndarray = 0.2 * self.__pre_goal_pos + 0.8 * self.__orienting_pos
                OpenGLUtil.render_cylinder(self.__pre_goal_pos, join_pos, 0.05, 0.05, slices=10)
                OpenGLUtil.render_cylinder(join_pos, self.__orienting_pos, 0.15, 0.0, slices=10)

    def terminate(self) -> None:
        """Tell the controller to terminate."""
        for inner_controller in self.__inner_controllers:
            inner_controller.terminate()

    # PRIVATE METHODS

    def __clear_goal_orientation(self) -> None:
        """Reset the variables associated with specifying a goal orientation."""
        self.__goal_orientation_valid = False
        self.__orienting_pos = None
        self.__pre_goal_pos = None

    def __clear_inner_controllers(self) -> None:
        """Clear the inner controllers queue."""
        # Stop the drone itself.
        self.__drone.stop()

        # Terminate all existing inner controllers.
        for inner_controller in self.__inner_controllers:
            inner_controller.terminate()

        # Clear the inner controllers queue.
        self.__inner_controllers = deque()

    def __get_active_inner_controller(self) -> Optional[DroneController]:
        """
        Get the active inner controller (if any).

        .. note::
            This may remove inner controllers that have finished in the process of trying to find the active one.

        :return:    The active inner controller (if any), or None otherwise.
        """
        active_inner_controller: Optional[DroneController] = None

        # If we haven't yet found an active inner controller, and there are still some to search through:
        while active_inner_controller is None and len(self.__inner_controllers) > 0:
            # Get the first inner controller in the queue.
            first_inner_controller: DroneController = self.__inner_controllers[0]

            # If it has finished, terminate it and remove it from the queue.
            if first_inner_controller.has_finished():
                first_inner_controller.terminate()
                self.__inner_controllers.popleft()

            # Otherwise, set the active inner controller to this controller.
            else:
                active_inner_controller = first_inner_controller

        return active_inner_controller

    def __get_last_inner_controller(self) -> Optional[DroneController]:
        """
        Get the last inner controller (if any).

        :return:    The last inner controller (if any), or None otherwise.
        """
        return self.__inner_controllers[-1] if len(self.__inner_controllers) > 0 else None

    def __render_traversability_sphere(self, pos: np.ndarray, *, radius: float) -> None:
        """
        Render a wireframe sphere at the specified position with a colour that indicates the traversability
        of the containing node.

        :param pos:     The specified position.
        :param radius:  The radius to use for the sphere.
        """
        node: PathNode = self.__planning_toolkit.pos_to_node(pos)
        if self.__planning_toolkit.node_is_traversable(node, use_clearance=True):
            glColor3f(0, 1, 0)
        elif self.__planning_toolkit.node_is_traversable(node, use_clearance=False):
            glColor3f(1, 0.5, 0)
        else:
            glColor3f(1, 0, 0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        OpenGLUtil.render_sphere(pos, radius, slices=10, stacks=10)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def __try_append_new_inner_controller(self, event: pygame.event.Event,
                                          last_inner_controller: Optional[DroneController]) -> None:
        """
        Try to append a new inner controller to the queue, as instructed by a PyGame event.

        .. note::
            For certain controllers, the last controller in the queue may be reused if possible.

        :param event:                   The PyGame event.
        :param last_inner_controller:   The most recent inner controller in the existing queue.
        """
        expected_drone_pos: Optional[np.ndarray] = last_inner_controller.get_expected_end_pos()
        expected_drone_state: Optional[Drone.EState] = last_inner_controller.get_expected_end_state()
        new_controller: Optional[DroneController] = None

        # If the user is releasing the left mouse button, and a goal position has been determined:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.__goal_pos is not None:
            # If the expected drone state once the previous controller finishes is either unknown or 'flying':
            if expected_drone_state is None or expected_drone_state == Drone.FLYING:
                # If either no goal orientation has been specified, or the one that has been specified is valid:
                if self.__pre_goal_pos is None or self.__goal_orientation_valid:
                    # If the last inner controller is a traverse waypoints one, reuse it, else construct a new one.
                    # noinspection PyUnusedLocal
                    traverse_waypoints_controller: Optional[TraverseWaypointsDroneController] = None
                    if type(last_inner_controller) is TraverseWaypointsDroneController:
                        traverse_waypoints_controller = cast(TraverseWaypointsDroneController, last_inner_controller)
                    else:
                        traverse_waypoints_controller = TraverseWaypointsDroneController(
                            debug=self.__debug, drone=self.__drone, planning_toolkit=self.__planning_toolkit
                        )

                        # If we do construct a new controller, record that, as it will need to be appended to the queue.
                        new_controller = traverse_waypoints_controller

                    # Append the relevant waypoints to the controller.
                    waypoints: List[np.ndarray] = []
                    if self.__pre_goal_pos is not None:
                        waypoints.append(self.__pre_goal_pos)
                        waypoints.append(self.__goal_pos)
                    else:
                        waypoints.append(self.__goal_pos)

                    traverse_waypoints_controller.append_waypoints(waypoints)

        # Otherwise, if the user is clicking the right mouse button:
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            # If either the expected drone state when the previous controller finishes is known to be 'flying',
            # or the expected drone state is unknown and one of the Ctrl keys is being pressed:
            if (expected_drone_state is not None and expected_drone_state == Drone.FLYING) or (
                expected_drone_state is None and pygame.key.get_mods() & pygame.KMOD_CTRL
            ):
                # If the expected drone position when the previous controller finishes is unknown, or it's known
                # and there's a flat patch of ground below it:
                if expected_drone_pos is None or \
                        self.__planning_toolkit.find_flat_ground_below(expected_drone_pos) is not None:
                    # Make a new landing controller.
                    new_controller = LandingDroneController(
                        drone=self.__drone, planning_toolkit=self.__planning_toolkit
                    )

            # Otherwise, if the expected drone state when the previous controller finishes is either unknown or 'idle':
            elif expected_drone_state is None or expected_drone_state == Drone.IDLE:
                # Make a new takeoff controller.
                new_controller = TakeoffDroneController(drone=self.__drone)

        # If a new controller has been constructed:
        if new_controller is not None:
            # Set its estimated start position as the estimated end position of the previous controller.
            new_controller.set_expected_start_pos(expected_drone_pos)

            # Append the newly constructed controller to the queue of inner controllers.
            self.__inner_controllers.append(new_controller)

    def __try_set_new_inner_controller(self, event: pygame.event.Event, drone_pos: np.ndarray) -> None:
        """
        Try to clear any existing inner controllers and add a new one to the queue, as instructed by a PyGame event.

        :param event:       The PyGame event.
        :param drone_pos:   The current position of the drone.
        """
        drone_state: Optional[Drone.EState] = self.__drone.get_state()
        new_controller: Optional[DroneController] = None

        # If the user is releasing the left mouse button, and a goal position has been determined:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.__goal_pos is not None:
            # If the current drone state is either unknown or 'flying':
            if drone_state is None or drone_state == Drone.FLYING:
                # If either no goal orientation has been specified, or the one that has been specified is valid:
                if self.__pre_goal_pos is None or self.__goal_orientation_valid:
                    # Make a traverse waypoints controller.
                    traverse_waypoints_controller: TraverseWaypointsDroneController = TraverseWaypointsDroneController(
                        debug=self.__debug, drone=self.__drone, planning_toolkit=self.__planning_toolkit
                    )

                    # Set the waypoints of the controller.
                    waypoints: List[np.ndarray] = []
                    if self.__pre_goal_pos is not None:
                        waypoints.append(self.__pre_goal_pos)
                        waypoints.append(self.__goal_pos)
                    else:
                        waypoints.append(self.__goal_pos)

                    traverse_waypoints_controller.set_waypoints(waypoints)

                    # Record that a new controller has been constructed.
                    new_controller = traverse_waypoints_controller

        # Otherwise, if the user is clicking the right mouse button:
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            # If either the current drone state is known to be 'flying', or it's unknown and one of the Ctrl keys
            # is being pressed:
            if (drone_state is not None and drone_state == Drone.FLYING) or (
                drone_state is None and pygame.key.get_mods() & pygame.KMOD_CTRL
            ):
                # If there's a flat patch of ground below the current drone position:
                if self.__planning_toolkit.find_flat_ground_below(drone_pos) is not None:
                    # Make a new landing controller.
                    new_controller = LandingDroneController(
                        drone=self.__drone, planning_toolkit=self.__planning_toolkit
                    )

            # Otherwise, if the current drone state is either unknown or 'idle':
            elif drone_state is None or drone_state == Drone.IDLE:
                # Make a new takeoff controller.
                new_controller = TakeoffDroneController(drone=self.__drone)

        # If a new controller has been constructed:
        if new_controller is not None:
            # Clear the inner controllers queue.
            self.__clear_inner_controllers()

            # Replace the queue of inner controllers with a singleton queue containing only the new controller.
            self.__inner_controllers = deque([new_controller])

    def __update_goal(self) -> None:
        """Update the goal that the user wants the drone to achieve."""
        # Pick from the viewing pose, and try to use the position of the mouse to determine:
        #  (i) The picked position, a point on the ground in the scene (if any) that the user is directly designating.
        # (ii) The floating position, a point that's floating in space a specified height above the picked position.
        # FIXME: This currently assumes that there is a single picking sub-window that's at the top-left of the
        #        overall window. We should make this more general.
        picking_image, picking_mask = self.__picker.pick(
            np.linalg.inv(CameraPoseConverter.camera_to_pose(self.__viewing_camera))
        )

        mx, my = pygame.mouse.get_pos()

        floating_pos: Optional[np.ndarray] = None
        picked_pos: Optional[np.ndarray] = None

        # noinspection PyChainedComparisons
        if 0 <= mx < picking_mask.shape[1] and 0 <= my < picking_mask.shape[0] and picking_mask[my, mx] != 0:
            picked_pos = picking_image[my, mx]
            picked_pos = self.__planning_toolkit.pos_to_vpos(picked_pos)
            floating_pos = picked_pos + np.array([0, -self.__height_offset, 0])

        # If the left mouse button is pressed:
        if pygame.mouse.get_pressed(num_buttons=3)[0]:
            # Set the flag indicating that it's pressed.
            self.__left_mouse_down = True

            # Initialise a flag that will record whether or not we've been able to determine a goal orientation.
            orientation_determined: bool = False

            # If the floating position (see above) has been determined, and there's an existing goal position:
            if floating_pos is not None and self.__goal_pos is not None:
                # Project the vector from the goal position to the floating position into the horizontal plane
                # to try to determine the goal orientation vector.
                orientation: np.ndarray = floating_pos - self.__goal_pos
                orientation[1] = 0.0

                # If the goal orientation vector is a sufficient length, indicating that the user has moved the
                # mouse away from the original ground position and wants to specify a goal orientation:
                orientation_length: float = np.linalg.norm(orientation)
                if orientation_length >= 0.2:
                    # Normalize the goal orientation vector, record that we've been able to determine a
                    # goal orientation, and set the associated variables accordingly.
                    orientation = vg.normalize(orientation)
                    orientation_determined = True
                    self.__orienting_pos = self.__goal_pos + 0.5 * orientation
                    self.__pre_goal_pos = self.__goal_pos - 1.0 * orientation
                    self.__goal_orientation_valid = self.__planning_toolkit.line_segment_is_traversable(
                        self.__planning_toolkit.pos_to_vpos(self.__pre_goal_pos),
                        self.__planning_toolkit.pos_to_vpos(self.__goal_pos),
                        use_clearance=True
                    )

            # If we haven't been able to determine a goal orientation, reset the associated variables accordingly.
            if not orientation_determined:
                self.__clear_goal_orientation()

        # Otherwise, if the left mouse button has just been released, reset the flag to False, but leave the
        # variables specifying the goal alone (since they'll be used later in the frame).
        elif self.__left_mouse_down:
            self.__left_mouse_down = False

        # Otherwise, set the goal and ground positions to be the floating and picked positions, respectively,
        # and reset the variables associated with specifying a goal orientation.
        else:
            self.__goal_pos = floating_pos
            self.__ground_pos = picked_pos
            self.__clear_goal_orientation()
