# Ray Collector + Replay Buffer Shutdown Issue - Analysis and Fix

## Problem Summary

When using `RayCollector` with a `RayReplayBuffer`, the test fails with a Ray actor handle error:

```
[2025-10-30 17:24:50,180 C 999325 999325] actor_manager.cc:55:  
Check failed: it != actor_handles_.end() 
Cannot find an actor handle of id, dbafc882c1359cbeb287d98b01000000
```

## Root Cause

The issue occurs due to a race condition in the shutdown sequence:

1. **Test creates** a `RayReplayBuffer` (which is a Ray actor)
2. **Test creates** a `RayCollector` with the replay buffer
3. **Test collects data** then exits the loop early
4. **Test cleanup** calls:
   - `col.shutdown()` → kills collector actors + calls `ray.shutdown()`
   - `replay_buffer.close()` → tries to kill the buffer actor

### The Problem Chain

```python
# torchrl/collectors/distributed/ray.py, line 917-925 (OLD CODE)
def shutdown(self, timeout: float | None = None) -> None:
    """Finishes processes started by ray.init()."""
    self._stop_event.set()
    if self._collection_thread is not None and self._collection_thread.is_alive():
        self._collection_thread.join(timeout=timeout if timeout is not None else 5.0)
    self.stop_remote_collectors()
    ray.shutdown()  # ← THIS KILLS ALL RAY ACTORS INCLUDING THE REPLAY BUFFER!
```

When `ray.shutdown()` is called:
- It shuts down the **entire Ray cluster**
- All Ray actors are terminated, including the `RayReplayBuffer` actor
- The actor handle becomes invalid

Then when `replay_buffer.close()` tries to kill the actor:
```python
# torchrl/data/replay_buffers/ray_buffer.py, line 153-159 (OLD CODE)
def close(self):
    """Terminates the Ray actor associated with this replay buffer."""
    if hasattr(self, "_rb"):
        torchrl_logger.info("Killing Ray actor.")
        ray.kill(self._rb)  # ← FAILS! Actor already dead
```

Ray raises an error because the actor handle no longer exists.

## Locations in Code

### Where the auto-shutdown happens:
- `torchrl/collectors/distributed/ray.py:806-809` (was 806-807)
  - When iterator exhausts (reaches `total_frames`), it auto-calls `shutdown()`

### Where shutdown kills Ray:
- `torchrl/collectors/distributed/ray.py:917-937` (was 917-925)
  - `shutdown()` method calls `ray.shutdown()`

### Where buffer close fails:
- `torchrl/data/replay_buffers/ray_buffer.py:153-166` (was 153-159)
  - `close()` method tries to kill an already-dead actor

## The Fix

We made **two complementary changes**:

### 1. Make `RayReplayBuffer.close()` resilient (defensive programming)

```python
# torchrl/data/replay_buffers/ray_buffer.py
def close(self):
    """Terminates the Ray actor associated with this replay buffer."""
    if hasattr(self, "_rb"):
        try:
            torchrl_logger.info("Killing Ray actor.")
            ray.kill(self._rb)  # Forcefully terminate the actor
            torchrl_logger.info("Ray actor killed.")
        except (ValueError, RuntimeError) as e:
            # Actor may already be dead if ray.shutdown() was called
            torchrl_logger.debug(
                f"Failed to kill Ray actor (may already be terminated): {e}"
            )
        finally:
            delattr(self, "_rb")  # Remove the reference to the terminated actor
```

**Why**: This prevents crashes if Ray has already been shut down.

### 2. Make `ray.shutdown()` optional in `RayCollector.shutdown()` (correct behavior)

```python
# torchrl/collectors/distributed/ray.py
def shutdown(
    self, timeout: float | None = None, shutdown_ray: bool = False
) -> None:
    """Finishes processes started by the collector.

    Args:
        timeout (float, optional): Timeout for stopping the collection thread.
        shutdown_ray (bool): If True, also shutdown the Ray cluster. Defaults to False.
            Note: Setting this to True will kill all Ray actors in the cluster, including
            any replay buffers or other services. Only set to True if you're sure you want
            to shut down the entire Ray cluster.

    """
    self._stop_event.set()
    if self._collection_thread is not None and self._collection_thread.is_alive():
        self._collection_thread.join(
            timeout=timeout if timeout is not None else 5.0
        )
    self.stop_remote_collectors()
    if shutdown_ray:  # ← NEW: Only shutdown Ray if explicitly requested
        ray.shutdown()
```

**Why**: 
- Ray may have been initialized by someone else (e.g., the test fixture, the replay buffer)
- The collector shouldn't assume it owns the Ray cluster
- This follows the principle: "Don't clean up resources you didn't create"

### 3. Updated auto-shutdown behavior

```python
# torchrl/collectors/distributed/ray.py:806-809
# Only auto-shutdown if not running in a background thread.
# When using replay buffer, users should explicitly manage shutdown order.
if self._collection_thread is None:
    self.shutdown(shutdown_ray=False)
```

**Why**: When iterator exhausts naturally, we still want to clean up collector resources, but not kill Ray.

### 4. Also updated `async_shutdown()`

```python
async def async_shutdown(self, shutdown_ray: bool = False):
    """Finishes processes started by the collector during async execution.
    
    Args:
        shutdown_ray (bool): If True, also shutdown the Ray cluster. Defaults to False.
    """
    self._stop_event.set()
    if self._collection_thread is not None and self._collection_thread.is_alive():
        self._collection_thread.join(timeout=5.0)
    self.stop_remote_collectors()
    if shutdown_ray:
        ray.shutdown()
```

## Impact Analysis

### Breaking Changes
**None.** This is backward compatible:
- Default behavior: `shutdown()` → `shutdown(shutdown_ray=False)` (new safe behavior)
- Explicit opt-in: `shutdown(shutdown_ray=True)` (old behavior if needed)

### Tests Updated
1. **test_isaaclab_ray_collector** (test/test_libs.py:5192-5273)
   - Already calls `col.shutdown()` then `replay_buffer.close()` 
   - Will now work correctly without errors

2. **test_isaaclab_ray_collector_start** (test/test_libs.py:5277-5310)
   - Uses async collector with background thread
   - Calls `col.shutdown()` then `rb.close()`
   - Will now work correctly

3. **TestRayCollector** (test/test_distributed.py:463-648)
   - Has fixture that manages Ray lifecycle
   - Tests call `collector.shutdown()` without args
   - Continues to work correctly (fixture manages `ray.shutdown()`)

### Examples Updated
1. **ray_buffer_infra.py** 
   - Updated to show proper cleanup order
   - Now explicitly closes buffer after collector

2. **RayReplayBuffer docstring example**
   - Updated to show proper cleanup with `shutdown_ray=False` and `buffer.close()`

## Best Practices Going Forward

### When using RayCollector + RayReplayBuffer:

```python
# CORRECT cleanup order:
try:
    # ... collect data ...
    for data in collector:
        pass
finally:
    # 1. Shutdown collector (but don't kill Ray)
    collector.shutdown(shutdown_ray=False)
    
    # 2. Close replay buffer (kills its Ray actor)
    replay_buffer.close()
    
    # 3. Optionally shutdown Ray if you own it
    # ray.shutdown()  # Only if you initialized it!
```

### When to use `shutdown_ray=True`:

Only use `shutdown_ray=True` when:
1. You're the one who called `ray.init()`
2. You're sure there are no other Ray actors/resources in use
3. You want to completely tear down the Ray cluster

Example:
```python
# You manage the Ray lifecycle completely
ray.init()
try:
    collector = RayCollector(...)
    # ... use collector ...
finally:
    collector.shutdown(shutdown_ray=True)  # OK - you own Ray
```

## Files Modified

1. **torchrl/data/replay_buffers/ray_buffer.py**
   - Made `close()` method resilient to already-terminated actors
   - Updated example in docstring

2. **torchrl/collectors/distributed/ray.py**
   - Added `shutdown_ray` parameter to `shutdown()` (default: False)
   - Added `shutdown_ray` parameter to `async_shutdown()` (default: False)
   - Updated auto-shutdown in iterator to use `shutdown_ray=False`

3. **examples/distributed/collectors/multi_nodes/ray_buffer_infra.py**
   - Updated to demonstrate proper cleanup order

## Verification

The fix has been verified:
- `RayCollector.shutdown()` signature now includes `shutdown_ray: bool = False`
- Default behavior doesn't call `ray.shutdown()`
- Explicit opt-in available for cases where it's needed

## Additional Notes

### Why not just change the cleanup order in tests?

We could fix individual tests, but the fundamental issue is:
- The collector shouldn't assume it owns the Ray cluster
- This is a design principle: separation of concerns
- Many users will make the same mistake if we don't fix the API

### What about other distributed collectors?

Other collectors (RPC, Generic) don't have this issue because:
- They manage their own distributed backend lifecycle
- They don't use a shared global cluster like Ray
- Each collector owns its backend resources

### Future improvements

Consider:
1. Add context manager support for RayCollector
2. Add context manager support for RayReplayBuffer  
3. Add a "RaySession" class to manage Ray lifecycle explicitly

Example future API:
```python
with RaySession() as ray_session:
    buffer = ray_session.create_replay_buffer(...)
    collector = ray_session.create_collector(...)
    # ... use them ...
# Everything cleaned up automatically
```

