from typing import Optional, List, Tuple

import numpy as np
import pygame

from OpenGL.GL import *
from typing import Any, Dict

from smg.navigation import Path, PathNode, PlanningToolkit
from smg.opengl import OpenGLUtil
from smg.pyoctomap import OctomapPicker, OcTree
from smg.rigging.cameras import Camera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController
from .traverse_waypoints_drone_controller import TraverseWaypointsDroneController


class RTSStyleDroneController(DroneController):
    """An RTS("real-time strategy")-style flight controller for a drone."""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, picker: OctomapPicker,
                 planning_octree: OcTree, viewing_camera: Camera):
        """
        Construct an RTS-style flight controller for a drone.

        :param debug:           Whether to enable debugging.
        :param drone:           The drone.
        :param picker:          A picker for the scene octree.
        :param planning_octree: The planning octree (used for path planning).
        :param viewing_camera:  The virtual camera being used to view the scene.
        """
        self.__goal_pos: Optional[np.ndarray] = None
        self.__height_offset: float = 0.5
        self.__inner_controller: TraverseWaypointsDroneController = TraverseWaypointsDroneController(
            debug=debug, drone=drone, planning_octree=planning_octree
        )
        # noinspection PyTypeChecker
        self.__picker: OctomapPicker = picker
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
        # TODO
        kwargs: Dict[str, Any] = {key: value for key, value in locals().items() if key != "self"}

        # Pick from the viewing pose.
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

        # If no PyGame events were passed in, use an empty list of events as the default.
        if events is None:
            events = []

        # Process any PyGame events that have happened since the last iteration.
        for event in events:
            # TODO
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.__goal_pos is not None:
                # TODO
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.__inner_controller.append_waypoints([self.__goal_pos])

                # TODO
                else:
                    self.__inner_controller.set_waypoints([self.__goal_pos])

            # TODO
            elif event.type == pygame.MOUSEWHEEL:
                self.__height_offset = np.clip(self.__height_offset + event.y * 0.2, 0.3, 3.0)

        # TODO
        self.__inner_controller.iterate(**kwargs)

    def render_ui(self) -> None:
        """Render the user interface for the controller."""
        # TODO
        path: Optional[Path] = self.__inner_controller.get_path()
        if path is not None:
            path.render(
                start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                waypoint_colourer=self.__inner_controller.get_occupancy_colourer()
            )

        if self.__goal_pos is not None:
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
