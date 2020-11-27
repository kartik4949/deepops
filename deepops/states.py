from enum import Enum


class TensorState(Enum):
    """Tensor State which tells the current state of Tensor i.e
    Detach, Device, Host.
    """

    DEVICE = "DEVICE"
    HOST = "HOST"
    DETACH = "DETACH"
