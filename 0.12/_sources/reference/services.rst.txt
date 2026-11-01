.. currentmodule:: torchrl

Service Registry
================

.. _ref_services:

TorchRL provides a service registry system for managing distributed services across workers in distributed applications. 
This is particularly useful for sharing resources like tokenizers, replay buffers, or Python executor pools across 
multiple environments or collectors.

The service registry provides a **backend-agnostic API** for distributed service management. While the current 
implementation focuses on Ray as the primary backend, the design allows for future backends (e.g., Monarch, 
local multiprocessing) without changing the core API.

Overview
--------

The service registry provides a centralized way to register and access distributed services that can be shared across 
different parts of your application. Services are registered once and can be accessed by any worker, with the underlying 
backend handling the distributed communication and resource management.

**Current Backend Support:**

- **Ray**: Full support for Ray-based distributed services (recommended for production use)
- **Other backends**: Planned for future releases (e.g., Monarch, local multiprocessing)

Key Features
~~~~~~~~~~~~

- **Centralized Management**: Register services once and access them from anywhere in your distributed system
- **Namespace Isolation**: Services are isolated within namespaces for multi-tenant support
- **Type Safety**: Dict-like access with ``services["name"]`` syntax
- **Automatic Cleanup**: Reset all services in a namespace with a single call
- **Backend Flexibility**: Designed to support multiple distributed backends (currently Ray)

Basic Usage
-----------

Getting Started
~~~~~~~~~~~~~~~

The service registry API is backend-agnostic, but you need to specify which backend to use when getting the registry.
Currently, Ray is the only supported backend.

.. code-block:: python

    import ray
    from torchrl.services import get_services
    
    # Initialize your backend (Ray in this example)
    ray.init()
    
    # Get the service registry for your chosen backend
    services = get_services(backend="ray", namespace="my_namespace")
    
    # Register a service (the class will become a distributed service)
    services.register(
        "tokenizer",
        TokenizerService,
        vocab_size=50000,
        num_cpus=1,  # Backend-specific option (Ray)
    )
    
    # Access the service from any worker
    # (other workers just need to call get_services with the same backend and namespace)
    services = get_services(backend="ray", namespace="my_namespace")
    tokenizer = services["tokenizer"]
    
    # Call the service (syntax depends on backend)
    # For Ray, you need to use .remote() and ray.get()
    result = ray.get(tokenizer.encode.remote("Hello world"))
    
    # Cleanup when done
    services.reset()
    ray.shutdown()

Service Registration
~~~~~~~~~~~~~~~~~~~~

Services are registered by providing a name, a class (that will become a distributed service), and any initialization arguments.
The exact behavior depends on the backend being used.

**Basic Registration (Backend-Agnostic):**

.. code-block:: python

    # Register a service with constructor arguments
    services.register(
        "my_service",
        MyServiceClass,
        arg1="value1",
        arg2="value2",
    )

The ``register`` method accepts:

- **name** (str): Unique identifier for the service
- **service_factory** (type): Class to instantiate as a distributed service
- **kwargs**: Arguments passed to the service constructor and/or backend-specific options

**Backend-Specific Options (Ray):**

When using the Ray backend, you can pass Ray actor options alongside constructor arguments:

.. code-block:: python

    # Ray-specific: Mix actor options and constructor arguments
    services.register(
        "gpu_service",
        GPUService,
        model_name="gpt2",      # Constructor argument
        num_cpus=4,             # Ray actor option
        num_gpus=1,             # Ray actor option
        max_concurrency=16,     # Ray actor option
    )

For more explicit separation of backend options and constructor arguments, the Ray backend provides 
``register_with_options`` (note that options are expected not to collide with constructor arguments):

.. code-block:: python

    # Ray-specific: Explicit separation of options
    services.register_with_options(
        "my_service",
        MyServiceClass,
        actor_options={
            "num_cpus": 4,
            "num_gpus": 1,
            "max_concurrency": 16,
        },
        model_name="gpt2",  # Constructor argument
        batch_size=32,      # Constructor argument
    )

.. note::
    The ``register_with_options`` method is specific to the Ray backend. Other backends may have 
    different mechanisms for separating backend options from constructor arguments.

Service Access
~~~~~~~~~~~~~~

Services can be accessed using dict-like syntax:

.. code-block:: python

    # Check if service exists
    if "tokenizer" in services:
        tokenizer = services["tokenizer"]
    
    # Get service (raises KeyError if not found)
    tokenizer = services["tokenizer"]
    
    # Alternative: use get() method
    tokenizer = services.get("tokenizer")
    
    # List all services
    service_names = services.list()
    print(f"Available services: {service_names}")

Cross-Worker Visibility
~~~~~~~~~~~~~~~~~~~~~~~

Services registered by one worker are immediately visible to all other workers in the same namespace.
This is a core feature of the service registry, enabled by the underlying distributed backend.

**Example with Ray Backend:**

.. code-block:: python

    import ray
    from torchrl.services import get_services
    
    @ray.remote
    class Worker:
        def register_service(self):
            # Worker 1: Register a service
            services = get_services(backend="ray", namespace="shared")
            services.register("shared_tokenizer", TokenizerService, vocab_size=50000)
            return "registered"
        
        def use_service(self):
            # Worker 2: Use the service registered by Worker 1
            services = get_services(backend="ray", namespace="shared")
            tokenizer = services["shared_tokenizer"]
            return ray.get(tokenizer.encode.remote("Hello"))
    
    # Worker 1 registers the service
    worker1 = Worker.remote()
    ray.get(worker1.register_service.remote())
    
    # Worker 2 can immediately use it
    worker2 = Worker.remote()
    result = ray.get(worker2.use_service.remote())

The key insight is that both workers access the same service registry by using the same ``backend`` and 
``namespace`` parameters in ``get_services()``. The backend handles the distributed coordination.

Namespace Isolation
~~~~~~~~~~~~~~~~~~~

Different namespaces provide complete isolation between service registries:

.. code-block:: python

    # Training namespace
    train_services = get_services(backend="ray", namespace="training")
    train_services.register("tokenizer", TokenizerService, vocab_size=50000)
    
    # Evaluation namespace
    eval_services = get_services(backend="ray", namespace="evaluation")
    eval_services.register("tokenizer", TokenizerService, vocab_size=30000)
    
    # These are completely independent services
    assert "tokenizer" in train_services
    assert "tokenizer" in eval_services
    # But they have different configurations

Cleanup
~~~~~~~

Always clean up services when done to free resources:

.. code-block:: python

    # Reset all services in a namespace
    services.reset()
    
    # This terminates all service actors and clears the registry
    # After reset(), the registry is empty
    assert services.list() == []

Python Executor Service
-----------------------

One of the most useful built-in services is the :class:`~torchrl.envs.llm.transforms.PythonExecutorService`, 
which provides a shared pool of Python interpreter processes for executing code across multiple environments.
This service is designed to work with any backend, though it's currently optimized for Ray.

Overview
~~~~~~~~

The Python Executor Service allows you to share a fixed pool of Python interpreters (e.g., 32 processes) across 
many environments (e.g., 128 environments). This provides significant resource savings compared to giving each 
environment its own interpreter process. The service is registered through the service registry and can be 
accessed by any worker using the :class:`~torchrl.envs.llm.transforms.PythonInterpreter` transform.

**Resource Efficiency:**

+---------------------------+---------------+------------+------------------+
| Configuration             | Environments  | Processes  | Resource Usage   |
+===========================+===============+============+==================+
| Local (persistent)        | 128           | 128        | 100%             |
+---------------------------+---------------+------------+------------------+
| Service (pool=32)         | 128           | 32         | **25%**          |
+---------------------------+---------------+------------+------------------+
| Service (pool=64)         | 128           | 64         | **50%**          |
+---------------------------+---------------+------------+------------------+

Basic Usage
~~~~~~~~~~~

The Python Executor Service is registered like any other service, then accessed through the 
:class:`~torchrl.envs.llm.transforms.PythonInterpreter` transform by specifying ``services="ray"`` 
(or the appropriate backend name).

**Example with Ray Backend:**

.. code-block:: python

    import ray
    from torchrl.services import get_services
    from torchrl.envs.llm.transforms import PythonExecutorService, PythonInterpreter
    from torchrl.envs.llm import ChatEnv
    
    # Initialize your backend
    ray.init()
    
    # Register the Python executor service
    services = get_services(backend="ray", namespace="my_namespace")
    services.register(
        "python_executor",
        PythonExecutorService,
        pool_size=32,           # 32 interpreter processes
        timeout=10.0,           # 10 second timeout
        num_cpus=32,            # Ray-specific: Allocate 32 CPUs
        max_concurrency=32,     # Ray-specific: Allow 32 concurrent executions
    )
    
    # Create environments that use the service
    env = ChatEnv(
        batch_size=(128,),  # 128 parallel environments
        system_prompt="Execute Python code when requested.",
    )
    
    # Add PythonInterpreter transform configured to use the service
    env = env.append_transform(
        PythonInterpreter(
            services="ray",              # Use Ray backend
            namespace="my_namespace",    # Same namespace as registration
        )
    )
    
    # All 128 environments now share the 32 interpreters!
    # The backend (Ray) automatically queues requests when all interpreters are busy

Optional Service Usage
~~~~~~~~~~~~~~~~~~~~~~

The :class:`~torchrl.envs.llm.transforms.PythonInterpreter` transform supports optional service usage. 
You can easily switch between using a shared service or local processes:

.. code-block:: python

    # Option 1: Use shared Ray service (recommended for many envs)
    env = env.append_transform(
        PythonInterpreter(
            services="ray",
            namespace="my_namespace",
        )
    )
    
    # Option 2: Use local persistent processes (good for few envs)
    env = env.append_transform(
        PythonInterpreter(
            services=None,
            persistent=True,
        )
    )
    
    # Option 3: Use temporary processes (good for infrequent use)
    env = env.append_transform(
        PythonInterpreter(
            services=None,
            persistent=False,
        )
    )

Conditional Usage Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

You can decide at runtime whether to use a distributed service based on your configuration:

.. code-block:: python

    import ray
    from torchrl.services import get_services
    from torchrl.envs.llm.transforms import PythonExecutorService, PythonInterpreter
    
    num_envs = 128
    use_distributed_service = ray.is_initialized() and num_envs > 16
    
    if use_distributed_service:
        # Use distributed service for efficient resource usage
        services = get_services(backend="ray")  # Could be other backends in the future
        if "python_executor" not in services:
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=32,
                timeout=10.0,
                num_cpus=32,            # Backend-specific option
                max_concurrency=32,     # Backend-specific option
            )
        
        # Configure transform to use the service
        interpreter = PythonInterpreter(services="ray")
    else:
        # Use local processes (no distributed backend)
        interpreter = PythonInterpreter(services=None, persistent=True)
    
    env = env.append_transform(interpreter)

How It Works
~~~~~~~~~~~~

The Python Executor Service uses a simple round-robin assignment strategy to distribute work across 
a pool of interpreter processes. The backend handles concurrency control and request queuing.

**Architecture:**

1. **Pool of Interpreters**: The service maintains a fixed pool of ``PersistentPythonProcess`` instances
2. **Round-Robin Assignment**: Each request is assigned to the next interpreter in the pool
3. **Backend Queuing**: When all interpreters are busy, the backend queues additional requests
4. **Concurrent Execution**: The backend controls how many requests can execute simultaneously

.. code-block:: python

    # Inside PythonExecutorService
    def execute(self, code: str) -> dict:
        # Simple round-robin assignment
        with self._lock:
            process = self.processes[self.next_idx]
            self.next_idx = (self.next_idx + 1) % self.pool_size
        
        # Backend handles queuing (e.g., Ray's max_concurrency parameter)
        return process.execute(code)

**Backend-Specific Behavior:**

- **Ray**: Uses the ``max_concurrency`` parameter to control concurrent executions. Requests beyond 
  this limit are automatically queued by Ray's actor system.
- **Other backends**: Will have their own mechanisms for concurrency control and queuing.

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to Use Service Mode (Distributed):**

- Running > 16 parallel environments
- Resource efficiency is important
- Code execution is frequent
- Have a distributed backend available (e.g., Ray)

**When to Use Local Persistent Mode:**

- Running < 16 environments
- Need strict isolation between environments
- Latency is critical
- Don't want distributed backend dependency

**When to Use Local Temp File Mode:**

- Code execution is infrequent
- Don't want persistent processes
- Memory is more important than speed

Advanced Usage
--------------

Multiple Service Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can register multiple services with different configurations:

.. code-block:: python

    services = get_services(backend="ray")
    
    # Fast service for simple code
    services.register(
        "python_executor_fast",
        PythonExecutorService,
        pool_size=16,
        timeout=5.0,
        num_cpus=16,
        max_concurrency=16,
    )
    
    # Heavy service for complex code
    services.register(
        "python_executor_heavy",
        PythonExecutorService,
        pool_size=64,
        timeout=30.0,
        num_cpus=64,
        max_concurrency=64,
    )
    
    # Use different services for different environments
    fast_env = env.append_transform(
        PythonInterpreter(services="ray", service_name="python_executor_fast")
    )
    heavy_env = env.append_transform(
        PythonInterpreter(services="ray", service_name="python_executor_heavy")
    )

Custom Services
~~~~~~~~~~~~~~~

You can create your own services by defining a class and registering it:

.. code-block:: python

    class MyCustomService:
        """A custom service for your application."""
        
        def __init__(self, config: dict):
            self.config = config
            # Initialize your service
        
        def process(self, data: str) -> dict:
            # Process data and return results
            return {"result": f"Processed: {data}"}
    
    # Register the custom service
    services = get_services(backend="ray")
    services.register(
        "my_service",
        MyCustomService,
        config={"param1": "value1"},
        num_cpus=2,
    )
    
    # Use the service
    my_service = services["my_service"]
    result = ray.get(my_service.process.remote("Hello"))

API Reference
-------------

Service Registry
~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.services

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    get_services
    ServiceBase
    RayService

Python Executor Service
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.envs.llm.transforms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    PythonExecutorService
    PythonInterpreter

Best Practices
--------------

1. **Specify Backend and Namespace**: Always explicitly specify both the backend and namespace when calling 
   ``get_services()`` to ensure services are registered and accessed from the correct location.

2. **Clean Up**: Always call ``services.reset()`` when done to free resources and terminate distributed services.

3. **Service Naming**: Use descriptive names that indicate the service's purpose (e.g., ``"python_executor"``, 
   ``"tokenizer_service"``).

4. **Backend-Specific Options**: Understand which options are backend-specific (e.g., ``num_cpus``, ``num_gpus``, 
   ``max_concurrency`` for Ray) and which are constructor arguments for your service class.

5. **Error Handling**: Check if services exist before accessing them:

   .. code-block:: python

       if "my_service" in services:
           service = services["my_service"]
       else:
           # Register or handle missing service

6. **Conditional Registration**: Only register services if they don't already exist:

   .. code-block:: python

       if "python_executor" not in services:
           services.register("python_executor", PythonExecutorService, ...)

7. **Context Managers**: Consider using context managers for automatic cleanup:

   .. code-block:: python

       class ServiceContext:
           def __init__(self, backend, namespace):
               self.services = get_services(backend=backend, namespace=namespace)
           
           def __enter__(self):
               return self.services
           
           def __exit__(self, *args):
               self.services.reset()
       
       with ServiceContext("ray", "my_namespace") as services:
           services.register("my_service", MyService)
           # Use services...
       # Automatic cleanup

8. **Backend Portability**: When writing code that should work with multiple backends, avoid using 
   backend-specific methods like ``register_with_options()`` (Ray-only). Stick to the common ``register()`` 
   API for maximum portability.

Examples
--------

For complete examples, see:

- ``examples/services/distributed_services.py`` - Basic service registry usage
- ``examples/llm/python_executor_service.py`` - Python executor service examples
- ``test/test_services.py`` - Comprehensive test suite
- ``test/test_python_executor_service.py`` - Python executor service tests

See Also
--------

- :doc:`llms` - LLM API documentation
- :ref:`ref_collectors` - Collector documentation
- `Ray Documentation <https://docs.ray.io/>`_ - Ray distributed framework documentation

.. note::
    **Future Backend Support**
    
    The service registry is designed to be backend-agnostic. While Ray is currently the only supported 
    backend, the API is structured to easily accommodate additional backends in the future, such as:
    
    - **Monarch**: For specialized distributed computing scenarios
    - **Local Multiprocessing**: For single-node parallelism without external dependencies
    - **Custom Backends**: You can implement your own backend by subclassing :class:`~torchrl.services.ServiceBase`
    
    The core API (``get_services()``, ``register()``, ``get()``, ``list()``, ``reset()``) will remain 
    consistent across all backends, ensuring your code remains portable.
