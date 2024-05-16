# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Headless GPU-accelerated OpenGL context creation on Google Colaboratory.
Typical usage:
    # Optional PyOpenGL configuratiopn can be done here.
    # import OpenGL
    # OpenGL.ERROR_CHECKING = True
    # 'glcontext' must be imported before any OpenGL.* API.
    from lucid.misc.gl.glcontext import create_opengl_context
    # Now it's safe to import OpenGL and EGL functions
    import OpenGL.GL as gl
    # create_opengl_context() creates a GL context that is attached to an
    # offscreen surface of the specified size. Note that rendering to buffers
    # of other sizes and formats is still possible with OpenGL Framebuffers.
    #
    # Users are expected to directly use the EGL API in case more advanced
    # context management is required.
    width, height = 640, 480
    create_opengl_context((width, height))
    # OpenGL context is available here.
"""

from __future__ import print_function

# pylint: disable=unused-import,g-import-not-at-top,g-statement-before-imports

try:
    import OpenGL  # noqa: F401
except ImportError:
    print("This module depends on PyOpenGL.")
    print(
        'Please run "\033[1m!pip install -q pyopengl\033[0m" '
        "prior importing this module."
    )
    raise

import ctypes
import os
from ctypes import pointer, util

os.environ["PYOPENGL_PLATFORM"] = "egl"

# OpenGL loading workaround.
#
# * PyOpenGL tries to load libGL, but we need libOpenGL, see [1,2].
#   This could have been solved by a symlink libGL->libOpenGL, but:
#
# * Python 2.7 can't find libGL and linEGL due to a bug (see [3])
#   in ctypes.util, that was only wixed in Python 3.6.
#
# So, the only solution I've found is to monkeypatch ctypes.util
# [1] https://devblogs.nvidia.com/egl-eye-opengl-visualization-without-x-server/
# [2] https://devblogs.nvidia.com/linking-opengl-server-side-rendering/
# [3] https://bugs.python.org/issue9998
_find_library_old = ctypes.util.find_library
try:

    def _find_library_new(name):
        return {
            "GL": "libOpenGL.so",
            "EGL": "libEGL.so",
        }.get(name, _find_library_old(name))

    util.find_library = _find_library_new
    import OpenGL.EGL as egl
    import OpenGL.GL as gl  # noqa: F401
    from OpenGL import error
    from OpenGL.EGL.EXT.device_base import egl_get_devices
    from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
except ImportError:
    print("Unable to load OpenGL libraries. " "Make sure you use GPU-enabled backend.")
    print(
        'Press "Runtime->Change runtime type" and set ' '"Hardware accelerator" to GPU.'
    )
    raise
finally:
    util.find_library = _find_library_old


def create_initialized_headless_egl_display():
    """Creates an initialized EGL display directly on a device."""
    for device in egl_get_devices():
        display = egl.eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, None)

        if display != egl.EGL_NO_DISPLAY and egl.eglGetError() == egl.EGL_SUCCESS:
            # `eglInitialize` may or may not raise an exception on failure depending
            # on how PyOpenGL is configured. We therefore catch a `GLError` and also
            # manually check the output of `eglGetError()` here.
            try:
                initialized = egl.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == egl.EGL_TRUE and egl.eglGetError() == egl.EGL_SUCCESS:
                    return display
    return egl.EGL_NO_DISPLAY


def create_opengl_context(surface_size=(640, 480)):
    """Create offscreen OpenGL context and make it current.
    Users are expected to directly use EGL API in case more advanced
    context management is required.
    Args:
      surface_size: (width, height), size of the offscreen rendering surface.
    """
    egl_display = create_initialized_headless_egl_display()
    if egl_display == egl.EGL_NO_DISPLAY:
        raise ImportError("Cannot initialize a headless EGL display.")

    major, minor = egl.EGLint(), egl.EGLint()
    egl.eglInitialize(egl_display, pointer(major), pointer(minor))

    config_attribs = [
        egl.EGL_SURFACE_TYPE,
        egl.EGL_PBUFFER_BIT,
        egl.EGL_BLUE_SIZE,
        8,
        egl.EGL_GREEN_SIZE,
        8,
        egl.EGL_RED_SIZE,
        8,
        egl.EGL_DEPTH_SIZE,
        24,
        egl.EGL_RENDERABLE_TYPE,
        egl.EGL_OPENGL_BIT,
        egl.EGL_NONE,
    ]
    config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

    num_configs = egl.EGLint()
    egl_cfg = egl.EGLConfig()
    egl.eglChooseConfig(
        egl_display, config_attribs, pointer(egl_cfg), 1, pointer(num_configs)
    )

    width, height = surface_size
    pbuffer_attribs = [
        egl.EGL_WIDTH,
        width,
        egl.EGL_HEIGHT,
        height,
        egl.EGL_NONE,
    ]
    pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
    egl_surf = egl.eglCreatePbufferSurface(egl_display, egl_cfg, pbuffer_attribs)

    egl.eglBindAPI(egl.EGL_OPENGL_API)

    egl_context = egl.eglCreateContext(egl_display, egl_cfg, egl.EGL_NO_CONTEXT, None)
    egl.eglMakeCurrent(egl_display, egl_surf, egl_surf, egl_context)
