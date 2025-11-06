"""Tests for PythonExecutorService with Ray service registry."""

import pytest

# Skip all tests if Ray is not available
pytest.importorskip("ray")

import ray
from torchrl.envs.llm.transforms import PythonExecutorService, PythonInterpreter
from torchrl.services import get_services


@pytest.fixture
def ray_init():
    """Initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init()
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestPythonExecutorService:
    """Test suite for PythonExecutorService."""

    def test_service_initialization(self, ray_init):
        """Test that the service can be created and registered."""
        namespace = "test_executor_init"
        services = get_services(backend="ray", namespace=namespace)

        try:
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=2,
                timeout=5.0,
                num_cpus=2,
                max_concurrency=2,
            )

            # Verify it was registered
            assert "python_executor" in services

            # Get the service
            executor = services["python_executor"]
            assert executor is not None

        finally:
            services.reset()

    def test_service_execution(self, ray_init):
        """Test that the service can execute Python code."""
        namespace = "test_executor_exec"
        services = get_services(backend="ray", namespace=namespace)

        try:
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=2,
                timeout=5.0,
                num_cpus=2,
                max_concurrency=2,
            )

            executor = services["python_executor"]

            # Execute simple code
            code = """
x = 10
y = 20
result = x + y
print(f"Result: {result}")
"""
            result = ray.get(executor.execute.remote(code), timeout=2)

            assert result["success"] is True
            assert "Result: 30" in result["stdout"]
            assert result["returncode"] == 0

        finally:
            services.reset()

    def test_service_execution_error(self, ray_init):
        """Test that the service handles execution errors."""
        namespace = "test_executor_error"
        services = get_services(backend="ray", namespace=namespace)

        try:
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=2,
                timeout=5.0,
                num_cpus=2,
                max_concurrency=2,
            )

            executor = services["python_executor"]

            # Execute code with an error
            code = "raise ValueError('Test error')"
            result = ray.get(executor.execute.remote(code), timeout=2)

            assert result["success"] is False
            assert "ValueError: Test error" in result["stderr"]

        finally:
            services.reset()

    def test_multiple_executions(self, ray_init):
        """Test multiple concurrent executions."""
        namespace = "test_executor_multi"
        services = get_services(backend="ray", namespace=namespace)

        try:
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=4,
                timeout=5.0,
                num_cpus=4,
                max_concurrency=4,
            )

            executor = services["python_executor"]

            # Submit multiple executions
            futures = []
            for i in range(8):
                code = f"print('Execution {i}')"
                futures.append(executor.execute.remote(code))

            # Wait for all to complete
            results = ray.get(futures, timeout=5)

            # All should succeed
            assert len(results) == 8
            for i, result in enumerate(results):
                assert result["success"] is True
                assert f"Execution {i}" in result["stdout"]

        finally:
            services.reset()


class TestPythonInterpreterWithService:
    """Test suite for PythonInterpreter using the service."""

    def test_interpreter_with_service(self, ray_init):
        """Test that PythonInterpreter can use the service."""
        namespace = "test_interp_service"
        services = get_services(backend="ray", namespace=namespace)

        try:
            # Register service
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=2,
                timeout=5.0,
                num_cpus=2,
                max_concurrency=2,
            )

            # Create interpreter with service
            interpreter = PythonInterpreter(
                services="ray",
                service_name="python_executor",
                namespace=namespace,
            )

            # Verify it's using the service
            assert interpreter.python_service is not None
            assert interpreter.processes is None
            assert interpreter.services == "ray"

        finally:
            services.reset()

    def test_interpreter_without_service(self):
        """Test that PythonInterpreter works without service."""
        # Create interpreter without service
        interpreter = PythonInterpreter(
            services=None,
            persistent=True,
        )

        # Verify it's using local processes
        assert interpreter.python_service is None
        assert interpreter.processes is not None
        assert interpreter.services is None

    def test_interpreter_execution_with_service(self, ray_init):
        """Test code execution through interpreter with service."""
        namespace = "test_interp_exec"
        services = get_services(backend="ray", namespace=namespace)

        try:
            # Register service
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=2,
                timeout=5.0,
                num_cpus=2,
                max_concurrency=2,
            )

            # Create interpreter with service
            interpreter = PythonInterpreter(services="ray", namespace=namespace)

            # Execute code
            code = "print('Hello from service')"
            result = interpreter._execute_python_code(code, 0)

            assert result["success"] is True
            assert "Hello from service" in result["stdout"]

        finally:
            services.reset()

    def test_interpreter_clone_preserves_service(self, ray_init):
        """Test that cloning an interpreter preserves service settings."""
        namespace = "test_interp_clone"
        services = get_services(backend="ray", namespace=namespace)

        try:
            # Register service
            services.register(
                "python_executor",
                PythonExecutorService,
                pool_size=2,
                timeout=5.0,
                num_cpus=2,
                max_concurrency=2,
            )

            # Create interpreter with service
            interpreter1 = PythonInterpreter(
                services="ray",
                service_name="python_executor",
                namespace=namespace,
            )

            # Clone it
            interpreter2 = interpreter1.clone()

            # Verify clone has same settings
            assert interpreter2.services == "ray"
            assert interpreter2.service_name == "python_executor"
            assert interpreter2.python_service is not None

        finally:
            services.reset()

    def test_interpreter_invalid_service_backend(self):
        """Test that invalid service backend raises error."""
        with pytest.raises(ValueError, match="Invalid services backend"):
            PythonInterpreter(services="invalid")

    def test_interpreter_missing_service(self, ray_init):
        """Test that missing service raises error."""
        with pytest.raises(RuntimeError, match="Failed to get Ray service"):
            PythonInterpreter(services="ray", service_name="nonexistent_service")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
