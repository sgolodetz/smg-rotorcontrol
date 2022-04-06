from .controllers.drone_controller import DroneController
from .controllers.follow_waypoints_drone_controller import FollowWaypointsDroneController
from .controllers.futaba_t6k_drone_controller import FutabaT6KDroneController
from .controllers.keyboard_drone_controller import KeyboardDroneController


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
        if controller_type == "follow_waypoints":
            return FollowWaypointsDroneController(**kwargs)
        elif controller_type == "futaba_t6k":
            return FutabaT6KDroneController(**kwargs)
        elif controller_type == "keyboard":
            return KeyboardDroneController(**kwargs)
        else:
            raise ValueError(f"Unknown drone controller type: {controller_type}")
