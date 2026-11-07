# Minecraft (CraftGround) Guide

This guide covers running Minecraft-based RL environments in TorchRL through
[CraftGround](https://github.com/yhs0602/CraftGround): what you must know about
Minecraft's license before you start, how to install the toolchain, and how to
run headless.

## Minecraft ownership and licensing (read this first)

TorchRL does not ship, bundle, or redistribute Minecraft, and neither does
CraftGround. CraftGround ships a [Fabric](https://fabricmc.net/) mod together
with a Gradle project; on the first environment start, that Gradle project
downloads the Minecraft client and its assets directly from Mojang's servers
onto your machine and launches the game in offline mode.

- **Own the game.** Users are expected to own a valid Minecraft: Java Edition
license (a one-time purchase from [minecraft.net](https://www.minecraft.net),
roughly USD 30). The environment runs the game in offline mode so that no
account login is required during headless training; offline mode bypasses
authentication, not the requirement to own the game.
- **EULA.** Use of Minecraft is governed by the
[Minecraft End User License Agreement](https://www.minecraft.net/eula) and
Mojang's usage guidelines, including their restriction on commercial
exploitation of the game. Academic and research use of Minecraft-based RL
environments has a long public history (Project Malmo, MineRL, MineDojo,
CraftGround); users and their organizations remain responsible for reviewing
the EULA for their own use case, in particular for any commercial use.
- **Never redistribute game files.** Do not publish Docker images, CI caches,
or any other artifacts that contain the downloaded Minecraft client or its
assets. Every machine, including CI runners, must fetch Minecraft from
Mojang itself through the build tooling. Recipes (Dockerfiles, scripts) that
download the game at build/run time on the user's machine are fine; baked
images are not.
- **Library licensing.** CraftGround is a separate, optional dependency; no
CraftGround code is vendored or redistributed by TorchRL. At the time of
writing, CraftGround's repository ships a GPL-3.0 license file while its
package metadata reports MIT. Consult CraftGround's upstream licensing
information before distributing software that depends on it.

## Installation

CraftGround needs Python packages, a JDK, and a recent CMake:

```
conda create -n craftground python=3.11 -y
conda activate craftground
conda install -c conda-forge openjdk=21 cmake -y
pip install craftground
```

Notes:

- OpenJDK 21 is required (Minecraft 1.21 targets Java 21). Any distribution
works (`conda-forge`, `apt install openjdk-21-jdk`, ...); make sure
`JAVA_HOME` points at it if you have several JDKs installed.
- `pip install craftground` pulls `craftground-runtime-mc121`, which contains
the Fabric mod's Gradle project. The first `env.reset()` runs
`./gradlew runClient` in that project: Gradle downloads Minecraft from
Mojang, compiles the mod, and boots the game. Expect several minutes and a
few GB of downloads on the very first run; later runs reuse the caches
(`~/.gradle` and the runtime package directory).
- If you change `JAVA_HOME` or the native toolchain after a failed first run,
clear the stale CMake cache with
`python -c "import craftground; craftground.clear_native_build_cache()"`.

## Usage with TorchRL

```
from craftground.initial_environment_config import (
 InitialEnvironmentConfig,
 WorldType,
)
from torchrl.envs import CraftGroundEnv, StepCounter, TransformedEnv

env = CraftGroundEnv(
 initial_env_config=InitialEnvironmentConfig(
 image_width=114,
 image_height=64,
 world_type=WorldType.SUPERFLAT,
 ),
 port=8023,
)
env = TransformedEnv(env, StepCounter(max_steps=100))
td = env.rollout(10)
```

The base environment is a sandbox: it emits a zero reward and never
terminates. Compose rewards and termination conditions with TorchRL
transforms, and interact with the world by sending Minecraft commands through
`env.add_command("...")` (executed on the next step).

## Headless operation

The Minecraft client needs an X display and OpenGL 3.2+. On a headless Linux
machine the simplest setup is a virtual framebuffer with Mesa software
rendering:

```
apt-get install -y xvfb libgl1-mesa-dev libegl1-mesa-dev libglew-dev \
 libglu1-mesa-dev libglfw3-dev xorg-dev
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

Software rendering is sufficient for functional tests and small-frame
training. For GPU-accelerated rendering on headless NVIDIA machines, use
[VirtualGL](https://virtualgl.org/) and pass `use_vglrun=True` to the
environment constructor; see the
[CraftGround headless guide](https://yhs0602.github.io/CraftGround/installation/headless)
for the full VirtualGL/X server configuration.

## Troubleshooting

- **`gradlew` not found / permission denied**: the runtime package directory
may have been installed without execute bits; the wrapper fixes permissions
automatically, but a custom `env_path` must contain an executable `gradlew`.
- **First reset hangs**: check the Gradle build by constructing the
environment with `verbose_gradle=True` (through
`CraftGroundEnv(craftground_kwargs={"verbose_gradle": True}, ...)`); most
first-run failures are missing JDK, missing X display, or blocked network
access to Mojang's download servers.
- **Orphan Java processes** after a crash: CraftGround removes orphans on the
next start; `pkill -f runClient` cleans up manually.