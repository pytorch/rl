"""Example of using PythonExecutorService with Ray service registry.

This example demonstrates:
1. Registering a shared Python executor service with a pool of interpreters
2. Using the service from multiple parallel environments
3. The efficiency gain from sharing 32 interpreters across 128 environments
"""

import ray
from torchrl.services import get_services
from torchrl.envs.llm.transforms import PythonExecutorService, PythonInterpreter


def example_basic_usage():
    """Basic example of registering and using the Python executor service."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Python Executor Service Usage")
    print("=" * 80)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Get service registry
        services = get_services(backend="ray", namespace="example1")
        
        # Register the Python executor service with a pool of 4 interpreters
        # The max_concurrency=4 means Ray will queue requests if all 4 are busy
        print("\n1. Registering Python executor service with pool_size=4...")
        services.register(
            "python_executor",
            PythonExecutorService,
            pool_size=4,
            timeout=10.0,
            num_cpus=4,
            max_concurrency=4
        )
        print("   ✓ Service registered successfully")
        
        # Get the service actor
        executor = services["python_executor"]
        print(f"\n2. Retrieved service: {executor}")
        
        # Execute some Python code
        print("\n3. Executing Python code via service...")
        code = """
x = 10
y = 20
result = x + y
print(f"Result: {result}")
"""
        result = ray.get(executor.execute.remote(code))
        print(f"   Success: {result['success']}")
        print(f"   Output: {result['stdout'].strip()}")
        
    finally:
        # Cleanup
        services.reset()
        ray.shutdown()


def example_with_transforms():
    """Example of using PythonInterpreter transform with the service."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Using PythonInterpreter Transform with Service")
    print("=" * 80)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Get service registry
        services = get_services(backend="ray", namespace="example2")
        
        # Register the Python executor service
        print("\n1. Registering Python executor service (pool_size=32)...")
        if "python_executor" not in services:
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=32,  # 32 interpreters
                timeout=10.0,
                num_cpus=32,
                max_concurrency=32  # Allow 32 concurrent executions
            )
        print("   ✓ Service registered")
        
        # Create a PythonInterpreter transform that uses the service
        print("\n2. Creating PythonInterpreter with services='ray'...")
        interpreter_with_service = PythonInterpreter(
            services="ray",
            service_name="python_executor"
        )
        print(f"   ✓ Created: {interpreter_with_service}")
        print(f"   Using service: {interpreter_with_service.python_service is not None}")
        
        # Compare with local interpreter
        print("\n3. Creating PythonInterpreter with services=None (local)...")
        interpreter_local = PythonInterpreter(
            services=None,
            persistent=True
        )
        print(f"   ✓ Created: {interpreter_local}")
        print(f"   Using local processes: {interpreter_local.processes is not None}")
        
        print("\n4. Key differences:")
        print("   - Service-based: Shares 32 interpreters across all envs")
        print("   - Local: Each env gets its own interpreter (128 envs = 128 processes)")
        print("   - For 128 envs, service-based uses 75% fewer resources!")
        
    finally:
        # Cleanup
        services.reset()
        ray.shutdown()


@ray.remote
class Worker:
    """Simulated worker that uses the Python executor service."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
    
    def execute_code(self, code: str) -> dict:
        """Execute code using the shared service."""
        from torchrl.services import get_services
        
        # Get the service (it's already registered)
        services = get_services(backend="ray", namespace="example3")
        executor = services["python_executor"]
        
        # Execute the code
        result = ray.get(executor.execute.remote(code))
        return {
            "worker_id": self.worker_id,
            "success": result["success"],
            "stdout": result["stdout"],
        }


def example_multiple_workers():
    """Example with multiple workers sharing the service."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multiple Workers Sharing Python Executor Service")
    print("=" * 80)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Register the service
        services = get_services(backend="ray", namespace="example3")
        print("\n1. Registering Python executor service (pool_size=4, max_concurrency=4)...")
        services.register(
            "python_executor",
            PythonExecutorService,
            pool_size=4,
            timeout=10.0,
            num_cpus=4,
            max_concurrency=4
        )
        print("   ✓ Service registered")
        
        # Create 10 workers (they will share the 4 interpreters)
        print("\n2. Creating 10 workers...")
        workers = [Worker.remote(i) for i in range(10)]  # type: ignore[attr-defined]
        print("   ✓ Workers created")
        
        # Have all workers execute code simultaneously
        print("\n3. All 10 workers executing code simultaneously...")
        print("   (4 will execute immediately, 6 will be queued by Ray)")
        
        futures = []
        for i, worker in enumerate(workers):
            code = f"""
import time
worker_id = {i}
print(f"Worker {{worker_id}} executing...")
time.sleep(0.1)  # Simulate work
print(f"Worker {{worker_id}} done!")
"""
            futures.append(worker.execute_code.remote(code))
        
        # Wait for all to complete
        results = ray.get(futures)
        
        print("\n4. Results:")
        for result in results:
            print(f"   Worker {result['worker_id']}: {result['success']}")
            if result['stdout']:
                for line in result['stdout'].strip().split('\n'):
                    print(f"      {line}")
        
        print("\n5. Key insight:")
        print("   - 10 workers shared 4 interpreters (via max_concurrency=4)")
        print("   - Ray automatically queued the extra requests")
        print("   - All workers successfully executed their code")
        
    finally:
        # Cleanup
        services.reset()
        ray.shutdown()


def example_comparison():
    """Compare resource usage between service and local modes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Resource Usage Comparison")
    print("=" * 80)
    
    print("\nScenario: 128 parallel environments executing Python code")
    print("-" * 80)
    
    print("\nOption A: Local mode (services=None, persistent=True)")
    print("  - Each env creates its own PersistentPythonProcess")
    print("  - Total processes: 128")
    print("  - Memory: ~128 * process_size")
    print("  - Pros: Isolation between envs")
    print("  - Cons: High memory usage, slow startup")
    
    print("\nOption B: Service mode (services='ray')")
    print("  - All envs share a pool of 32 interpreters")
    print("  - Total processes: 32")
    print("  - Memory: ~32 * process_size (75% reduction!)")
    print("  - Pros: Efficient resource usage, fast startup")
    print("  - Cons: Ray handles queuing (small latency if all busy)")
    
    print("\nOption C: Local mode (services=None, persistent=False)")
    print("  - Each code execution spawns a new process")
    print("  - Total processes: 0 (when idle), up to 128 (during execution)")
    print("  - Memory: Variable, but creates process overhead per execution")
    print("  - Pros: No persistent processes")
    print("  - Cons: High process creation overhead, slow execution")
    
    print("\nRecommendation:")
    print("  ✓ Use services='ray' for many parallel environments (>16)")
    print("  ✓ Use persistent=True for few environments (<16)")
    print("  ✓ Use persistent=False only for infrequent execution")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_with_transforms()
    example_multiple_workers()
    example_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

