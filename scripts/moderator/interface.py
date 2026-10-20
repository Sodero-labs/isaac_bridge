"""The command / state endpoint of the bridge: a ZMQ SUB socket for the gateways' commands and a PUB socket for
the state frames, both bound on this computer.

A receive thread queues every inbound frame; the physics thread decodes them at its next step
(Moderator.synchronize -> drain_inbound), in arrival order. Decoding in the receive thread would take the GIL away
from the physics step for every single command. The queue is bounded: nobody drains it while the timeline is
stopped, and stale commands are worthless anyway.
"""
from __future__ import annotations

import collections
import logging
import threading
from typing import TYPE_CHECKING

import zmq

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Tuple

INBOUND_MAXLEN = 4096
RECV_POLL_MS = 50
TOPIC_PREFIX = b"TM/"
DIVIDER = b"||"


class MessageHandler:
    def __init__(self, *, host: str, subscribe_port: int, publish_port: int):
        self.logger = logging.getLogger("isaac_bridge")
        self._host = host
        self._subscribe_port = subscribe_port
        self._publish_port = publish_port
        self._server_thread: "threading.Thread | None" = None
        self._flag_stop = False
        self._is_server_running = False
        self._context = None
        self._subscribe_socket = None
        self._publish_socket = None
        self.inbound = collections.deque(maxlen=INBOUND_MAXLEN)   # raw multipart frames, oldest first

    @property
    def is_server_running(self) -> bool:
        return self._is_server_running

    def drain_inbound(self, handler: "Callable[[List[bytes]], None]", limit: int = INBOUND_MAXLEN) -> int:
        """Hand the frames received since the last drain to `handler`, in arrival order, on the caller's thread.
        Returns the number of frames handled."""
        queue = self.inbound
        n = 0
        while queue and n < limit:
            data = queue.popleft()
            n += 1
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"inbound frame handling failed: {e}")
        return n

    def send(self, name: str, frames: "List[Tuple[str, bytes]]") -> None:
        """Publish b'TM/<name>/<data_type>||<payload>' for every (data_type, payload); never blocks (a frame that
        does not fit the send queue is dropped)."""
        if not self._is_server_running or not self._publish_socket:
            return
        prefix = TOPIC_PREFIX + name.encode() + b"/"
        for data_type, payload in frames:
            if payload is None:
                continue
            try:
                self._publish_socket.send(prefix + data_type.encode() + DIVIDER + payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                self.logger.warning(f"send failed for {name}: {e}")

    def start_server(self) -> bool:
        """Bind both sockets and start the receive thread. False when the ports cannot be bound."""
        if self._is_server_running:
            return True
        self._flag_stop = False
        try:
            self._context = zmq.Context()
            self._subscribe_socket = self._context.socket(zmq.SUB)
            self._subscribe_socket.setsockopt(zmq.RCVHWM, 500)
            self._subscribe_socket.bind(f"tcp://{self._host}:{self._subscribe_port}")
            self._subscribe_socket.setsockopt(zmq.SUBSCRIBE, TOPIC_PREFIX)
            self._publish_socket = self._context.socket(zmq.PUB)
            self._publish_socket.setsockopt(zmq.SNDHWM, 50)
            self._publish_socket.bind(f"tcp://{self._host}:{self._publish_port}")
            self._server_thread = threading.Thread(target=self._receive, daemon=True, name="IsaacBridgeZMQ")
            self._server_thread.start()
        except Exception as e:
            self.logger.error(f"could not bind ports {self._subscribe_port}/{self._publish_port}: {e}")
            self.stop_server()
            return False
        self._is_server_running = True
        self.logger.info(f"listening for commands on port {self._subscribe_port}, "
                         f"publishing state on port {self._publish_port}")
        return True

    def stop_server(self) -> bool:
        """Close the endpoint, never under a live receive thread.

        libzmq aborts the whole process when a socket is closed (or its context terminated) while another thread
        is inside recv on it, and that would take Isaac Sim down. The receive loop polls with a short timeout, so
        it normally exits well within the join below; if it is somehow still alive, the context is leaked instead
        of closed (the ports then stay bound until Isaac Sim exits).
        """
        self._flag_stop = True
        self._is_server_running = False
        thread = self._server_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        if thread is not None and thread.is_alive():
            self.logger.error("receive thread did not exit within 3 s; leaving the ZMQ context open rather than "
                              "risk aborting Isaac Sim (ports stay bound until it exits)")
            self._subscribe_socket = self._publish_socket = self._context = self._server_thread = None
            return False
        try:
            if self._subscribe_socket:
                self._subscribe_socket.close(linger=0)
            if self._publish_socket:
                self._publish_socket.close(linger=0)
            if self._context:
                self._context.term()
        except Exception as e:
            self.logger.warning(f"socket cleanup: {e}")
        was_bound = self._server_thread is not None
        self._subscribe_socket = self._publish_socket = self._context = self._server_thread = None
        if was_bound:
            self.logger.info("command / state ports closed")
        return True

    def _receive(self) -> None:
        poller = zmq.Poller()
        poller.register(self._subscribe_socket, zmq.POLLIN)
        while not self._flag_stop:
            try:
                socks = dict(poller.poll(RECV_POLL_MS))
                if self._flag_stop:
                    break
                if socks.get(self._subscribe_socket) == zmq.POLLIN:
                    # NOBLOCK: poll() said readable; a blocking recv could sit inside libzmq while stop_server()
                    # closes the socket underneath it
                    self.inbound.append(self._subscribe_socket.recv_multipart(flags=zmq.NOBLOCK))
            except zmq.ZMQError as e:
                if not self._flag_stop:
                    self.logger.error(f"receive loop: {e}")
            except Exception as e:
                if not self._flag_stop:
                    self.logger.error(f"receive loop: {type(e).__name__}: {e}")
