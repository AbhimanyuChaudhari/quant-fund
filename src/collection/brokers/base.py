from abc import ABC, abstractmethod


class BaseBroker(ABC):
    """
    Abstract base class for all broker connections.
    Every broker must implement these methods.
    Collector doesn't care which broker — just calls these.
    """

    @abstractmethod
    def login(self):
        """Authenticate with broker."""
        pass

    @abstractmethod
    def get_active_symbols(self) -> list[dict]:
        """
        Return list of active futures contracts.
        Each dict must have:
          tradingsymbol, instrument_token, exchange,
          expiry, lot_size
        """
        pass

    @abstractmethod
    def start_websocket(self, on_tick, on_connect,
                        on_error, on_close,
                        on_reconnect, on_noreconnect):
        """
        Start WebSocket connection.
        Calls provided callbacks on events.
        """
        pass

    @abstractmethod
    def subscribe(self, tokens: list[int]):
        """Subscribe to instrument tokens."""
        pass

    @abstractmethod
    def stop(self):
        """Stop WebSocket connection."""
        pass