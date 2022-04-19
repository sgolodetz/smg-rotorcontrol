from typing import Optional, List, Tuple

import numpy as np
import pygame

from OpenGL.GL import *
from typing import Any, Dict

from smg.navigation import Path
from smg.opengl import OpenGLUtil
from smg.pyoctomap import OctomapPicker, OcTree
from smg.rigging.cameras import Camera
from smg.rigging.helpers import CameraPoseConverter
from smg.rotory.drones import Drone

from .drone_controller import DroneController
from .traverse_waypoints_drone_controller import TraverseWaypointsDroneController


class RTSStyleDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = False, drone: Drone, intrinsics: Tuple[float, float, float, float],
                 planning_octree: OcTree, scene_octree: OcTree, viewing_camera: Camera, window_size: Tuple[int, int]):
        self.__height_offset: float = 0.5
        self.__inner_controller: TraverseWaypointsDroneController = TraverseWaypointsDroneController(
            debug=debug, drone=drone, planning_octree=planning_octree
        )
        # noinspection PyTypeChecker
        self.__picker: OctomapPicker = OctomapPicker(scene_octree, *window_size, intrinsics)
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
        if picking_mask[my, mx] != 0:
            self.__picker_pos = picking_image[my, mx] + np.array([0, -self.__height_offset, 0])

            # TODO
            self.__picker_pos = self.__inner_controller.get_planning_toolkit().pos_to_vpos(self.__picker_pos)

        # If no PyGame events were passed in, use an empty list of events as the default.
        if events is None:
            events = []

        # Process any PyGame events that have happened since the last iteration.
        for event in events:
            # TODO
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.__picker_pos is not None:
                self.__inner_controller.set_waypoints([self.__picker_pos])

            # TODO
            elif event.type == pygame.MOUSEWHEEL:
                self.__height_offset = np.clip(self.__height_offset + event.y * 0.2, 0.3, 3.0)

        # TODO
        self.__inner_controller.iterate(**kwargs)

    def render_ui(self) -> None:
        # TODO
        # interpolated_path: Optional[Path] = self.__inner_controller.get_interpolated_path()
        path: Optional[Path] = self.__inner_controller.get_path()
        if path is not None:
            path.render(
                start_colour=(0, 1, 1), end_colour=(0, 1, 1), width=5,
                waypoint_colourer=self.__inner_controller.get_occupancy_colourer()
            )
            # interpolated_path.render(
            #     start_colour=(1, 1, 0), end_colour=(1, 0, 1), width=5,
            #     waypoint_colourer=None
            # )

        if self.__picker_pos is not None:
            glColor3f(0, 1, 0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            OpenGLUtil.render_sphere(self.__picker_pos, 0.1, slices=10, stacks=10)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
