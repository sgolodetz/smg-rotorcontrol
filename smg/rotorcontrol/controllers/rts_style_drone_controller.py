import numpy as np
import pygame

from OpenGL.GL import *
from typing import Any, cast, Dict, Optional, List, Tuple

from smg.navigation import Path, PathNode, PlanningToolkit
from smg.opengl import OpenGLUtil
from smg.pyoctomap import OctomapPicker
from smg.rigging.cameras import Camera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController
from .traverse_waypoints_drone_controller import TraverseWaypointsDroneController


class RTSStyleDroneController(DroneController):
    """An RTS("real-time strategy")-style flight controller for a drone."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, picker: Optional[OctomapPicker],
                 planning_toolkit: PlanningToolkit, viewing_camera: Camera):
        """
        Construct an RTS-style flight controller for a drone.

        :param debug:               Whether to enable debugging.
        :param drone:               The drone.
        :param picker:              A picker for the scene (required: an exception will be raised if this is None).
        :param planning_toolkit:    The planning toolkit (used for path planning).
        :param viewing_camera:      The virtual camera being used to view the scene.
        """
        if picker is None:
            raise RuntimeError("Error: An RTS-style drone controller requires a picker for the scene to be provided")

        self.__goal_pos: Optional[np.ndarray] = None
        self.__height_offset: float = 1.0
        self.__inner_controller: TraverseWaypointsDroneController = TraverseWaypointsDroneController(
            debug=debug, drone=drone, planning_toolkit=planning_toolkit
        )
        self.__picker: OctomapPicker = cast(OctomapPicker, picker)
        self.__picker_pos: Optional[np.ndarray] = None
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
            self.__picker_pos = self.__inner_controller.get_planning_toolkit().pos_to_vpos(self.__picker_pos)
            self.__goal_pos = self.__picker_pos + np.array([0, -self.__height_offset, 0])
        else:
            self.__picker_pos = None
            self.__goal_pos = None

        # Process any PyGame events that have happened since the last iteration.
        for event in events:
            # If the user clicks the left mouse button, and a goal position has been determined:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.__goal_pos is not None:
                # If the user is currently pressing the 'left shift' key, append the goal position to the existing
                # list of waypoints.
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.__inner_controller.append_waypoints([self.__goal_pos])

                # Otherwise, replace the list of waypoints with a singleton list containing the goal position.
                else:
                    self.__inner_controller.set_waypoints([self.__goal_pos])

            # If the user scrolls the mouse wheel, change the desired offset of the goal position above the floor.
            elif event.type == pygame.MOUSEWHEEL:
                self.__height_offset = np.clip(self.__height_offset + event.y * 0.2, 0.3, 3.0)

        # Delegate lower-level control of the drone to the inner controller.
        self.__inner_controller.iterate(**kwargs)

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # Render the path that the drone is following (if any).
        path: Optional[Path] = self.__inner_controller.get_path()
        if path is not None:
            path.render(
                start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                waypoint_colourer=self.__inner_controller.get_occupancy_colourer()
            )

        # Render any new waypoints for which a path has not yet been planned.
        # FIXME: This is currently a bit messy - it needs tidying up and moving somewhere more sensible.
        glColor3f(1, 1, 0)

        new_waypoints: List[np.ndarray] = self.__inner_controller.get_new_waypoints()
        waypoints: List[np.ndarray] = self.__inner_controller.get_waypoints()
        last_waypoint: np.ndarray = self.__inner_controller.get_current_pos()
        if path is not None and len(new_waypoints) != len(waypoints):
            last_waypoint = path[-1].position

        glLineWidth(5)
        for i in range(len(new_waypoints)):
            OpenGLUtil.render_sphere(new_waypoints[i], 0.1, slices=10, stacks=10)
            glBegin(GL_LINES)
            glVertex3f(*last_waypoint)
            glVertex3f(*new_waypoints[i])
            glEnd()
            last_waypoint = new_waypoints[i]
        glLineWidth(1)

        # If a goal position has been determined:
        if self.__goal_pos is not None:
            # Render a sphere at that position with a colour that indicates the traversability of the goal node.
            toolkit: PlanningToolkit = self.__inner_controller.get_planning_toolkit()
            goal_node: PathNode = toolkit.pos_to_node(self.__goal_pos)
            if toolkit.node_is_traversable(goal_node, use_clearance=True):
                glColor3f(0, 1, 0)
            elif toolkit.node_is_traversable(goal_node, use_clearance=False):
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
        self.__inner_controller.terminate()
