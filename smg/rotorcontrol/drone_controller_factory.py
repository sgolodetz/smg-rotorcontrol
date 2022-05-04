from .controllers.drone_controller import DroneController
from .controllers.futaba_t6k_drone_controller import FutabaT6KDroneController
from .controllers.keyboard_drone_controller import KeyboardDroneController
from .controllers.rts_style_drone_controller import RTSStyleDroneController
from .controllers.traverse_waypoints_drone_controller import TraverseWaypointsDroneController


class DroneControllerFactory:
    """Used to make drone controllers."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_drone_controller(controller_type: str, **kwargs) -> DroneController:
        """
        Make a drone controller of the specified type.

        :param controller_type: The type of drone controller to make.
        :param kwargs:          Any keyword arguments to pass to the drone controller constructor.
        :return:                The drone controller.
        """
        if controller_type == "futaba_t6k":
            return FutabaT6KDroneController(**kwargs)
        elif controller_type == "keyboard":
            return KeyboardDroneController(**kwargs)
        elif controller_type == "rts":
            return RTSStyleDroneController(**kwargs)
        elif controller_type == "traverse_waypoints":
            return TraverseWaypointsDroneController(**kwargs)
        else:
            raise ValueError(f"Unknown drone controller type: {controller_type}")
