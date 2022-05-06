import numpy as np
import pygame

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
        self.__height_offset: float = 1.0
        self.__inner_controllers: Deque[DroneController] = deque()
        self.__picker: OctomapPicker = cast(OctomapPicker, picker)
        self.__picker_pos: Optional[np.ndarray] = None
        self.__planning_toolkit: PlanningToolkit = cast(PlanningToolkit, planning_toolkit)
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
        :param tracker_c_t_i:       A transformation from initial camera space to current camera space, as estimated
                                    by any tracker that's running (optional). Note that if the tracker is a monocular
                                    one, the transformation will be non-metric.
        """
        # Make a copy of the keyword arguments that have been passed to the method, so that they can later be
        # forwarded on to the inner controller.
        kwargs: Dict[str, Any] = {key: value for key, value in locals().items() if key != "self"}

        # If no PyGame events were passed in, use an empty list of events as the default.
        if events is None:
            events = []

        # Pick from the viewing pose, and try to determine a goal position based on the position of the mouse.
        # FIXME: This currently assumes that there is a single picking sub-window that's at the top-left of the
        #        overall window. We should make this more general.
        picking_image, picking_mask = self.__picker.pick(
            np.linalg.inv(CameraPoseConverter.camera_to_pose(self.__viewing_camera))
        )

        mx, my = pygame.mouse.get_pos()

        # noinspection PyChainedComparisons
        if 0 <= mx < picking_mask.shape[1] and 0 <= my < picking_mask.shape[0] and picking_mask[my, mx] != 0:
            self.__picker_pos = picking_image[my, mx]
            self.__picker_pos = self.__planning_toolkit.pos_to_vpos(self.__picker_pos)
            self.__goal_pos = self.__picker_pos + np.array([0, -self.__height_offset, 0])
        else:
            self.__picker_pos = None
            self.__goal_pos = None

        # Get the active and last inner controllers (if any).
        active_inner_controller: Optional[DroneController] = self.__get_active_inner_controller()
        last_inner_controller: Optional[DroneController] = self.__get_last_inner_controller()

        # Process any PyGame events that have happened since the last iteration.
        for event in events:
            # If the user clicks the left mouse button, and a goal position has been determined:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.__goal_pos is not None:
                # If the user is currently pressing one of the shift keys:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # If the last inner controller is a traverse waypoints controller, reuse it.
                    # Otherwise, construct a new one.
                    traverse_waypoints_controller: Optional[TraverseWaypointsDroneController] = None
                    if type(last_inner_controller) is TraverseWaypointsDroneController:
                        traverse_waypoints_controller = cast(TraverseWaypointsDroneController, last_inner_controller)
                    else:
                        traverse_waypoints_controller = TraverseWaypointsDroneController(
                            debug=self.__debug, drone=self.__drone, planning_toolkit=self.__planning_toolkit
                        )

                    # Append the goal position to the traverse waypoint controller's existing list of waypoints.
                    traverse_waypoints_controller.append_waypoints([self.__goal_pos])

                    # If the last inner controller is not a traverse waypoints controller:
                    if type(last_inner_controller) is not TraverseWaypointsDroneController:
                        # TODO: Comment here.
                        if last_inner_controller is not None:
                            traverse_waypoints_controller.set_estimated_start_pos(
                                last_inner_controller.get_estimated_end_pos()
                            )

                        # Append the newly constructed controller to the queue of inner controllers.
                        self.__inner_controllers.append(traverse_waypoints_controller)

                # Otherwise, if the user is not currently pressing one of the shift keys:
                else:
                    # Clear the inner controllers queue.
                    self.__clear_inner_controllers()

                    # Make a new traverse waypoints controller and set its list of waypoints to be a singleton
                    # list containing the goal position.
                    traverse_waypoints_controller: TraverseWaypointsDroneController = TraverseWaypointsDroneController(
                        debug=self.__debug, drone=self.__drone, planning_toolkit=self.__planning_toolkit
                    )

                    traverse_waypoints_controller.set_waypoints([self.__goal_pos])

                    # Replace the queue of inner controllers with a singleton queue containing only the newly
                    # constructed controller.
                    active_inner_controller = traverse_waypoints_controller
                    self.__inner_controllers = deque([active_inner_controller])

            # If the user clicks the right mouse button:
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                # If the user is currently pressing one of the control keys, make a landing controller.
                # Otherwise, make a takeoff controller.
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    new_controller: DroneController = LandingDroneController(
                        drone=self.__drone, planning_toolkit=self.__planning_toolkit
                    )
                else:
                    new_controller: DroneController = TakeoffDroneController(drone=self.__drone)

                # If the user is currently pressing one of the shift keys:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # TODO: Comment here.
                    if last_inner_controller is not None:
                        new_controller.set_estimated_start_pos(last_inner_controller.get_estimated_end_pos())

                    # Append the newly constructed controller to the queue of inner controllers.
                    self.__inner_controllers.append(new_controller)

                # Otherwise, if the user is not currently pressing one of the shift keys:
                else:
                    # Clear the inner controllers queue.
                    self.__clear_inner_controllers()

                    # Replace the queue of inner controllers with a singleton queue containing only the newly
                    # constructed controller.
                    active_inner_controller = new_controller
                    self.__inner_controllers = deque([active_inner_controller])

            # If the user scrolls the mouse wheel, change the desired offset of the goal position above the floor.
            elif event.type == pygame.MOUSEWHEEL:
                self.__height_offset = np.clip(self.__height_offset + event.y * 0.2, 0.3, 3.0)

        # Delegate lower-level control of the drone to the active inner controller (if any). If the active controller
        # finishes, remove it from the queue and move on to the next one (if any).
        if active_inner_controller is not None:
            active_inner_controller.iterate(**kwargs)
            if active_inner_controller.has_finished():
                active_inner_controller.terminate()
                self.__inner_controllers.popleft()

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # Render the user interfaces for the inner controllers.
        for inner_controller in self.__inner_controllers:
            inner_controller.render_ui()

        # If a goal position has been determined:
        if self.__goal_pos is not None:
            # Render a sphere at that position with a colour that indicates the traversability of the goal node.
            goal_node: PathNode = self.__planning_toolkit.pos_to_node(self.__goal_pos)
            if self.__planning_toolkit.node_is_traversable(goal_node, use_clearance=True):
                glColor3f(0, 1, 0)
            elif self.__planning_toolkit.node_is_traversable(goal_node, use_clearance=False):
                glColor3f(1, 0.5, 0)
            else:
                glColor3f(1, 0, 0)

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            OpenGLUtil.render_sphere(self.__goal_pos, 0.1, slices=10, stacks=10)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Render a vertical line from the goal position to the closest point on the ground beneath it.
            # This makes it easier for the user to see where the goal lies in relation to the scene.
            glLineWidth(5)
            glColor3f(1, 0, 0)
            glBegin(GL_LINES)
            glVertex3f(*self.__picker_pos)
            glVertex3f(*self.__goal_pos)
            glEnd()
            glLineWidth(1)

    def terminate(self) -> None:
        """Tell the controller to terminate."""
        for inner_controller in self.__inner_controllers:
            inner_controller.terminate()

    # PRIVATE METHODS

    def __clear_inner_controllers(self) -> None:
        """Clear the inner controllers queue."""
        # Stop the drone itself.
        self.__drone.stop()

        # Terminate all existing inner controllers.
        for inner_controller in self.__inner_controllers:
            inner_controller.terminate()

    def __get_active_inner_controller(self) -> Optional[DroneController]:
        """
        Get the active inner controller (if any).

        :return:    The active inner controller (if any), or None otherwise.
        """
        return self.__inner_controllers[0] if len(self.__inner_controllers) > 0 else None

    def __get_last_inner_controller(self) -> Optional[DroneController]:
        """
        Get the last inner controller (if any).

        :return:    The last inner controller (if any), or None otherwise.
        """
        return self.__inner_controllers[-1] if len(self.__inner_controllers) > 0 else None
