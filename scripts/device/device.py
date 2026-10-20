from abc import ABC, abstractmethod
from typing import List, Tuple


class WrappedDevice(ABC):
    """A prim of the stage that the bridge serves, addressed on the wire by its prim name."""

    @abstractmethod
    def destroy(self) -> None:
        """Release the Isaac Sim handles (the device is unbound)."""

    @abstractmethod
    def update_target(self, data_type: str, data: bytes) -> None:
        """Take one inbound frame addressed to this device (TM/<name>/<data_type>)."""

    @abstractmethod
    def get_actual(self) -> List[Tuple[str, bytes]]:
        """The frames to publish after this physics step: [(data_type, payload)]."""

    @abstractmethod
    def synchronize(self) -> None:
        """One physics step: apply the commands in force and read the state."""
