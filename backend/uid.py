import time
import threading
from loguru import logger

logger.remove()

logger.add("user_statistics.log", rotation="10 MB", retention="7 days", level="INFO")


class UidManager:
    """
    UID Management
    - The system allows a maximum of `max_uid_count` active UIDs at the same time.
    - Each UID can only exist once (i.e., the same UID cannot be counted multiple times).
    - Implements an expiration mechanism to remove inactive UIDs.
    - Thread-safe.
    """

    def __init__(self, max_uid_count=100, uid_timeout_second=3600):
        """
        - max_uid_count: Maximum number of active UIDs allowed in the system.
        - uid_timeout: UID expiration time (in seconds); if not accessed within this time, the UID is removed.
        """
        self.max_uid_count = max_uid_count
        self.uid_timeout_second = uid_timeout_second

        # Stores UID information: {uid: last_access_time (float)}
        self._uid_info = {}

        self._lock = threading.Lock()

    def _join(self, uid: str) -> bool:
        """
        Attempts to add a UID.
        - If the UID already exists, updates the last access time and returns True.
        - If the UID does not exist, checks the global UID limit:
          - If the limit is reached, returns False.
          - If under the limit, adds the UID and returns True.
        """
        with self._lock:
            logger.info(f"User {uid} performed action: join")

            # Clean up expired UIDs before adding a new one
            self.__cleanup_expired_locked()

            # If UID already exists, update its access time
            if uid in self._uid_info:
                self._uid_info[uid]['last_time'] = time.time()
                return True
            else:
                # Check if global UID limit is reached
                if len(self._uid_info) >= self.max_uid_count:
                    return False
                # Add new UID
                self._uid_info[uid] = {'last_time': time.time()}
                return True

    def _leave(self, uid: str):
        """
        Removes a UID from the system.
        - If the UID exists, it will be deleted.
        """
        with self._lock:
            self._uid_info.pop(uid, None)

    def _exists(self, uid: str) -> bool:
        """
        Checks if a UID exists and is still valid.
        """
        with self._lock:
            return uid in self._uid_info

    def _update_uid_info(self, uid: str, user_info: dict = dict()):
        """ 
        Updates the last access time of a UID.
        """
        with self._lock:
            if uid in self._uid_info:
                self._uid_info[uid].update(user_info)
                self._uid_info[uid]['last_time'] = time.time()
        
        logger.info(f"User {uid} update uid info with {user_info}")

    def cleanup_expired(self):
        """
        Public method to remove expired UIDs (thread-safe).
        """
        with self._lock:
            self.__cleanup_expired_locked()

    def __cleanup_expired_locked(self):
        """
        Internal method to remove expired UIDs.
        - This method should be called within a lock.
        """
        now = time.time()
        expired = [uid for uid, info in self._uid_info.items() if now - info.get('last_time', 0) > self.uid_timeout_second]

        for uid in expired:
            logger.info(f"User {uid} reaches the maximum expiration time and will be removed.")
            del self._uid_info[uid]

    def process(self, uid: str, user_info: dict = dict()):
        """
        Arguments:
            - uid(str): user id
            - user_info(dict, optional): user information as a dict
        """

        if not self._exists(uid):
            if not self._join(uid):
                logger.info("UID {uid} reaches max count")
                return RuntimeError(f"UID {uid} reaches max count")

        else:
            # if uid exists, update access time
            self._update_uid_info(uid, user_info)

    def get_current_uids(self):
        """
        Get all available UID(for debug)
        """
        with self._lock:
            return list(self._uid_info.keys())
