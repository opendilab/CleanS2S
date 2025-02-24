import random
import time
import threading
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from uid import UidManager

def worker_thread(thread_id, uid_manager, possible_uids, operation_count=100):
    """
    Each thread performs operation_count random operations.
    """
    for _ in range(operation_count):
        uid = random.choice(possible_uids)
        
        # Randomly select an operation
        action = random.choice(["join", "exists", "update", "leave"])
        
        if action == "join":
            joined = uid_manager._join(uid)
        
        elif action == "exists":
            ex = uid_manager._exists(uid)
        
        elif action == "update":
            uid_manager._update_access_time(uid)
        
        elif action == "leave":
            uid_manager._leave(uid)
        
        # Randomly sleep for a short time to create disorder
        time.sleep(random.uniform(0, 0.01))

def test_uid_manager_concurrent():
    uid_manager = UidManager(max_uid_count=5, uid_timeout=3)
    
    # List of possible UIDs
    possible_uids = [f"user{i}" for i in range(10)]
    
    # Start multiple threads
    threads = []
    thread_count = 10
    for i in range(thread_count):
        t = threading.Thread(
            target=worker_thread, 
            args=(i, uid_manager, possible_uids, 100)
        )
        threads.append(t)
        t.start()
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    # Perform one final cleanup of expired UIDs
    uid_manager.cleanup_expired()
    
    # Check the results
    current_uids = uid_manager.get_current_uids()
    print("Final surviving UIDs: ", current_uids)
    print("Final UID count: ", len(current_uids))

    assert len(current_uids) <= uid_manager.max_uid_count, "UID count exceeded the limit!"

    print("All tests passed")

if __name__ == "__main__":
    test_uid_manager_concurrent()
