import etcd
import socket
import time
import logging
from threading import Thread, Event
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class LeaderElection:
    def __init__(
        self,
        service_name: str,
        ttl: int = 10,
        etcd_host: str = 'localhost',
        etcd_port: int = 2379
    ):
        self.service_name = service_name
        self.ttl = ttl
        self.node_id = f"{service_name}-{socket.gethostname()}"
        self.leader_key = f"/leader/{service_name}"
        self.client = etcd.Client(host=etcd_host, port=etcd_port)
        self.is_leader = False
        self.stop_event = Event()
        self.heartbeat_thread: Optional[Thread] = None
        self._on_leader_callback: Optional[Callable[[], None]] = None
        self._on_follower_callback: Optional[Callable[[], None]] = None

    def on_become_leader(self, callback: Callable[[], None]):
        """Set callback for when this node becomes leader"""
        self._on_leader_callback = callback

    def on_become_follower(self, callback: Callable[[], None]):
        """Set callback for when this node becomes follower"""
        self._on_follower_callback = callback

    def start(self):
        """Start the leader election process"""
        try:
            self.stop_event.clear()
            self._try_become_leader()
            self.heartbeat_thread = Thread(target=self._heartbeat_loop)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
            logger.info(f"Started leader election for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to start leader election: {str(e)}")
            raise

    def stop(self):
        """Stop the leader election process"""
        try:
            self.stop_event.set()
            if self.heartbeat_thread:
                self.heartbeat_thread.join()
            if self.is_leader:
                self._release_leadership()
            logger.info(f"Stopped leader election for {self.service_name}")
        except Exception as e:
            logger.error(f"Error stopping leader election: {str(e)}")
            raise

    def _try_become_leader(self):
        """Attempt to become the leader"""
        try:
            self.client.write(
                self.leader_key,
                self.node_id,
                ttl=self.ttl,
                prevExist=False
            )
            self.is_leader = True
            logger.info(f"Node {self.node_id} became leader")
            if self._on_leader_callback:
                self._on_leader_callback()
        except etcd.EtcdAlreadyExist:
            self.is_leader = False
            if self._on_follower_callback:
                self._on_follower_callback()
            logger.info(f"Node {self.node_id} is follower")
        except Exception as e:
            logger.error(f"Error in leader election: {str(e)}")
            self.is_leader = False

    def _heartbeat_loop(self):
        """Maintain leadership with heartbeat"""
        while not self.stop_event.is_set():
            try:
                if self.is_leader:
                    self.client.refresh(self.leader_key, ttl=self.ttl)
                    logger.debug(f"Leader heartbeat sent: {self.node_id}")
                else:
                    self._try_become_leader()
            except etcd.EtcdKeyNotFound:
                self._try_become_leader()
            except Exception as e:
                logger.error(f"Error in heartbeat: {str(e)}")
                self.is_leader = False
            time.sleep(self.ttl / 3)

    def _release_leadership(self):
        """Release leadership voluntarily"""
        try:
            if self.is_leader:
                self.client.delete(self.leader_key)
                self.is_leader = False
                logger.info(f"Node {self.node_id} released leadership")
        except Exception as e:
            logger.error(f"Error releasing leadership: {str(e)}")

    @property
    def is_leader_node(self) -> bool:
        """Check if this node is currently the leader"""
        return self.is_leader

# Example usage:
# def on_leader():
#     print("This node is now the leader")
#
# def on_follower():
#     print("This node is now a follower")
#
# election = LeaderElection("spacecraft-controller")
# election.on_become_leader(on_leader)
# election.on_become_follower(on_follower)
# election.start()
# try:
#     # Run your application
#     pass
# finally:
#     election.stop()
