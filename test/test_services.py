# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

pytest.importorskip("ray")

# Import from mocking_classes which is a proper module
import sys
from pathlib import Path

import ray
from ray.util.state import get_actor as get_actor_by_id
from torchrl._utils import logger

sys.path.insert(0, str(Path(__file__).parent))


from test_services_fixtures import SimpleService, TokenizerService
from torchrl.services import get_services, RayService


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Initialize Ray once for the entire test module."""
    import os

    if ray.is_initialized():
        ray.shutdown(raise_on_error=False)

    # Add test directory to Ray worker PYTHONPATH so they can import test_services_fixtures
    test_dir = os.path.dirname(os.path.abspath(__file__))

    ray.init(
        ignore_reinit_error=True,
        namespace="test_torchrl_services",
        runtime_env={"env_vars": {"PYTHONPATH": test_dir}},
    )
    yield
    # Cleanup once at the end
    ray.shutdown()


@pytest.fixture(scope="function", autouse=True)
def kill_all_actors():
    """Kill all actors after each test."""
    yield
    if not ray.is_initialized():
        return
    # list actors
    actors = ray.state.actors()
    if not actors:
        return
    for actor_id, info in actors.items():
        if info["State"] == "ALIVE":
            try:
                actor = get_actor_by_id(actor_id)
                ray.kill(actor, no_restart=True)
            except Exception as e:
                logger.warning(f"Error killing actor {actor_id}: {e}")


@pytest.fixture(scope="function", autouse=True)
def cleanup_services():
    """Clean up services after each test to avoid conflicts."""
    # Run the test
    yield

    # After test: attempt to get and reset any services that might exist
    # This prevents namespace pollution between tests
    try:
        # Each test uses a unique namespace, so this is mostly for safety
        pass
    except Exception:
        pass


class TestRayService:
    """Test suite for RayService."""

    def test_initialization(self):
        """Test that RayService initializes correctly."""
        services = RayService(namespace="test_init")
        try:
            assert isinstance(services, RayService)
            # shutdown the Ray service
        finally:
            services.shutdown(raise_on_error=False)

    def test_initialization_with_existing_ray(self):
        """Test RayService with already-initialized Ray."""
        # Ray is already initialized by fixture
        services = RayService(namespace="test_torchrl_services")
        try:
            assert isinstance(services, RayService)
        finally:
            services.shutdown(raise_on_error=False)

    def test_register_service(self):
        """Test registering a new service."""
        services = RayService(namespace="test_register")
        try:
            actor = services.register("simple", SimpleService, value=42)
            assert actor is not None

            # Verify we can call methods on the actor
            result = ray.get(actor.get_value.remote(), timeout=10)
            assert result == 42
        finally:
            services.shutdown(raise_on_error=False)

    def test_register_with_ray_options(self):
        """Test registering a service with Ray options."""
        services = RayService(namespace="test_options")
        try:
            actor = services.register(
                "tokenizer", TokenizerService, vocab_size=5000, num_cpus=1, num_gpus=0
            )

            # Verify the service works
            result = ray.get(actor.encode.remote("hello"), timeout=10)
            assert isinstance(result, list)
            assert len(result) == 5
        finally:
            services.shutdown(raise_on_error=False)

    def test_register_duplicate_raises(self):
        """Test that registering duplicate service raises ValueError."""
        services = RayService(namespace="test_duplicate")

        try:
            services.register("simple", SimpleService)

            with pytest.raises(ValueError, match="already exists"):
                services.register("simple", SimpleService)
        finally:
            services.shutdown(raise_on_error=False)

    def test_get_service(self):
        """Test retrieving a registered service."""
        services = RayService(namespace="test_get")

        try:
            # Register a service
            original_actor = services.register("simple", SimpleService, value=100)

            # Retrieve the same service
            retrieved_actor = services.get("simple")

            # Verify they reference the same actor
            result = ray.get(retrieved_actor.get_value.remote(), timeout=10)
            assert result == 100
        finally:
            services.shutdown(raise_on_error=False)

    def test_get_nonexistent_raises(self):
        """Test that getting a nonexistent service raises KeyError."""
        services = RayService(namespace="test_get_missing")
        try:
            with pytest.raises(KeyError, match="not found"):
                services.get("nonexistent")
        finally:
            services.shutdown(raise_on_error=False)

    def test_getitem_access(self):
        """Test dict-like access with []."""
        services = RayService(namespace="test_getitem")

        try:
            services.register("tokenizer", TokenizerService, vocab_size=100)

            # Access using dict syntax
            tokenizer = services["tokenizer"]
            result = ray.get(tokenizer.encode.remote("test"), timeout=10)
            assert isinstance(result, list)
        finally:
            services.shutdown(raise_on_error=False)

    def test_contains(self):
        """Test checking if service exists with 'in'."""
        services = RayService(namespace="test_contains")

        try:
            services.register("existing", SimpleService)
            assert "existing" in services
            assert "nonexistent" not in services
        finally:
            services.shutdown(raise_on_error=False)

    def test_list_services(self):
        """Test listing all registered services."""
        services = RayService(namespace="test_list")

        try:
            # Initially empty
            assert services.list() == []

            # Register multiple services
            services.register("service1", SimpleService, value=1)
            services.register("service2", SimpleService, value=2)
            services.register("service3", TokenizerService)

            service_names = services.list()
            assert sorted(service_names) == ["service1", "service2", "service3"]
        finally:
            services.shutdown(raise_on_error=False)

    def test_cross_worker_visibility(self):
        """Test that services registered by one worker are visible to another."""
        namespace = "test_cross_worker"

        # Worker 1: Register a service
        services1 = RayService(namespace=namespace)
        services1.register("shared_service", SimpleService, value=999)

        # Worker 2: Should see the same service
        services2 = RayService(namespace=namespace)
        assert "shared_service" in services2
        try:
            shared_actor = services2["shared_service"]
            result = ray.get(shared_actor.get_value.remote(), timeout=10)
            assert result == 999
        finally:
            services2.shutdown(raise_on_error=False)

    def test_namespace_isolation(self):
        """Test that different namespaces isolate services."""
        # Register in namespace A
        services_a = RayService(namespace="namespace_a")
        services_a.register("service", SimpleService, value=111)

        # Register different service with same name in namespace B
        services_b = RayService(namespace="namespace_b")
        services_b.register("service", SimpleService, value=222)

        try:
            # Verify they're isolated
            actor_a = services_a["service"]
            actor_b = services_b["service"]

            result_a = ray.get(actor_a.get_value.remote(), timeout=10)
            result_b = ray.get(actor_b.get_value.remote(), timeout=10)

            assert result_a == 111
            assert result_b == 222
        finally:
            services_a.shutdown(raise_on_error=False)
            services_b.shutdown(raise_on_error=False)

    def test_options_method(self):
        """Test the register_with_options() method for explicit configuration."""
        services = RayService(namespace="test_options_method")

        try:
            # Register with explicit actor options
            services.register_with_options(
                "simple",
                SimpleService,
                actor_options={"num_cpus": 1, "max_concurrency": 5},
                value=42,
            )

            # Verify it works
            actor = services["simple"]
            result = ray.get(actor.get_value.remote(), timeout=10)
            assert result == 42
        finally:
            services.shutdown(raise_on_error=False)

    def test_service_persistence(self):
        """Test that services persist across RayService instances."""
        namespace = "test_persistence"

        # Create first instance and register
        services1 = RayService(namespace=namespace)
        services1.register("persistent", SimpleService, value=777)

        # Create second instance
        services2 = RayService(namespace=namespace)

        try:
            # Should be able to access the service
            assert "persistent" in services2
            actor = services2["persistent"]
            result = ray.get(actor.get_value.remote(), timeout=10)
            assert result == 777
        finally:
            services1.shutdown(raise_on_error=False)
            services2.shutdown(raise_on_error=False)

    def test_setitem_registration(self):
        """Test registering with dict-like syntax."""
        services = RayService(namespace="test_setitem")

        try:
            # Register using setitem
            services["simple"] = SimpleService

            # Verify it was registered
            assert "simple" in services
        finally:
            services.shutdown(raise_on_error=False)

    def test_reset(self):
        """Test resetting the service registry."""
        services = RayService(namespace="test_reset")

        try:
            # Register multiple services
            services.register("service1", SimpleService, value=1)
            services.register("service2", SimpleService, value=2)
            services.register("service3", TokenizerService)

            # Verify they exist
            assert len(services.list()) == 3
            assert "service1" in services
            assert "service2" in services
            assert "service3" in services

            # Reset
            services.reset()

            # Verify all services are gone
            assert len(services.list()) == 0
            assert "service1" not in services
            assert "service2" not in services
            assert "service3" not in services
        finally:
            services.shutdown(raise_on_error=False)

    def test_reset_multiple_namespaces(self):
        """Test that reset only affects the specific namespace."""
        # Create services in different namespaces
        services_a = RayService(namespace="namespace_a")
        services_b = RayService(namespace="namespace_b")

        try:
            # Register services in both namespaces
            services_a.register("service", SimpleService, value=1)
            services_b.register("service", SimpleService, value=2)

            # Verify both exist
            assert "service" in services_a
            assert "service" in services_b

            # Reset namespace A
            services_a.reset()

            # Verify namespace A is empty but B is not
            assert "service" not in services_a
            assert "service" in services_b

            # Cleanup namespace B
            services_b.reset()
        finally:
            services_a.shutdown(raise_on_error=False)
            services_b.shutdown(raise_on_error=False)


class TestGetServices:
    """Test suite for get_services() function."""

    def test_get_services_ray(self):
        """Test get_services with ray backend."""
        services = get_services(backend="ray", namespace="test_get_services")
        try:
            assert isinstance(services, RayService)
        finally:
            services.shutdown(raise_on_error=False)

    def test_get_services_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            get_services(backend="invalid")

    def test_get_services_with_ray_config(self):
        """Test get_services with Ray configuration."""
        services = get_services(
            backend="ray",
            namespace="test_with_config",
        )
        try:
            assert isinstance(services, RayService)
        finally:
            services.shutdown(raise_on_error=False)


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_tokenizer_sharing(self):
        """Test sharing a tokenizer across workers."""
        namespace = "test_tokenizer_integration"

        # Setup: Register tokenizer
        services = get_services(backend="ray", namespace=namespace)
        try:
            services.register(
                "tokenizer", TokenizerService, vocab_size=10000, num_cpus=1
            )

            # Worker 1: Use tokenizer
            tokenizer = services["tokenizer"]
            encoded = ray.get(tokenizer.encode.remote("hello world"), timeout=10)
            assert isinstance(encoded, list)
            assert len(encoded) == 11  # "hello world" has 11 characters

            # Worker 2: Access same tokenizer
            services2 = get_services(backend="ray", namespace=namespace)
            tokenizer2 = services2["tokenizer"]
            encoded2 = ray.get(tokenizer2.encode.remote("hello world"), timeout=10)

            # Should produce same results
            assert encoded == encoded2
        finally:
            services.shutdown(raise_on_error=False)

    def test_stateful_service(self):
        """Test that services maintain state across calls."""
        services = RayService(namespace="test_stateful")

        services.register("counter", SimpleService, value=0)

        try:
            counter = services["counter"]

            # Modify state
            ray.get(counter.set_value.remote(10), timeout=10)

            # Access from "different worker"
            services2 = RayService(namespace="test_stateful")
            counter2 = services2["counter"]

            # Should see the modified state
            result = ray.get(counter2.get_value.remote(), timeout=10)
            assert result == 10
        finally:
            services.shutdown(raise_on_error=False)
            # services2.shutdown(raise_on_error=False)

    def test_conditional_registration(self):
        """Test pattern: register only if not exists."""
        namespace = "test_conditional"

        services1 = get_services(backend="ray", namespace=namespace)

        # Worker 2: Try to register same service
        services2 = get_services(backend="ray", namespace=namespace)

        try:
            # Worker 1: Register if not exists
            if "shared_tokenizer" not in services1:
                services1.register(
                    "shared_tokenizer", TokenizerService, vocab_size=5000
                )

            if "shared_tokenizer" not in services2:
                services2.register(
                    "shared_tokenizer", TokenizerService, vocab_size=10000
                )
            else:
                # Should take this branch
                pass

            # Both should see the same tokenizer (first one registered)
            tok1 = services1["shared_tokenizer"]
            tok2 = services2["shared_tokenizer"]

            # Verify they're the same actor
            vocab1 = ray.get(tok1.getattr.remote("vocab_size"), timeout=10)
            vocab2 = ray.get(tok2.getattr.remote("vocab_size"), timeout=10)
            assert vocab1 == vocab2 == 5000
        finally:
            services1.shutdown(raise_on_error=False)
            services2.shutdown(raise_on_error=False)

    def test_multiple_services_management(self):
        """Test managing multiple different services."""
        services = RayService(namespace="test_multiple")

        try:
            # Register various services
            services.register(
                "tokenizer", TokenizerService, vocab_size=1000, num_cpus=1
            )
            services.register("counter1", SimpleService, value=1, num_cpus=0.5)
            services.register("counter2", SimpleService, value=2, num_cpus=0.5)

            # Verify all are accessible
            assert len(services.list()) == 3

            tok = services["tokenizer"]
            c1 = services["counter1"]
            c2 = services["counter2"]

            # Use them
            assert ray.get(tok.getattr.remote("vocab_size"), timeout=10) == 1000
            assert ray.get(c1.getattr.remote("value"), timeout=10) == 1
            assert ray.get(c2.getattr.remote("value"), timeout=10) == 2
        finally:
            services.shutdown(raise_on_error=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _, args = parser.parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + args)
