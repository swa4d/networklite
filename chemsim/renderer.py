"""
renderer.py – Real-time 3D particle cloud renderer using OpenGL + GLFW.

Falls back gracefully if OpenGL dependencies are not installed.
The renderer runs in a separate thread to keep the simulation loop responsive.
"""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from chemsim.simulator import SimulationResult


# ─── OpenGL availability check ────────────────────────────────────────────────

def _opengl_available() -> bool:
    try:
        import glfw
        import OpenGL.GL as gl
        return True
    except ImportError:
        return False

# Note: imgui is intentionally NOT a dependency — the `imgui` PyPI package
# is incompatible with Python 3.14+ and is unmaintained. Parameter tuning
# overlays can be added via glfw callbacks without it.


OPENGL_AVAILABLE = _opengl_available()

# ─── Vertex shader ────────────────────────────────────────────────────────────

_VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;

uniform mat4 MVP;
out vec3 vColor;

void main() {
    gl_Position  = MVP * vec4(aPos, 1.0);
    gl_PointSize = clamp(aSize, 2.0, 30.0);
    vColor       = aColor;
}
"""

_FRAG_SHADER = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    // Circular point sprite
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    if (dot(cxy, cxy) > 1.0) discard;
    FragColor = vec4(vColor, 1.0);
}
"""

# ─── Color palette for species ────────────────────────────────────────────────

_SPECIES_COLORS = np.array([
    [0.00, 0.45, 0.70],  # blue
    [0.90, 0.62, 0.00],  # amber
    [0.00, 0.62, 0.45],  # green
    [0.84, 0.37, 0.00],  # vermillion
    [0.80, 0.47, 0.65],  # pink
    [0.35, 0.71, 0.92],  # sky blue
    [0.94, 0.89, 0.26],  # yellow
    [0.60, 0.60, 0.60],  # gray
], dtype=np.float32)


# ─── Particle layout ──────────────────────────────────────────────────────────

def _build_particle_array(
    concentrations: np.ndarray,  # [n_species]
    n_particles_total: int = 10_000,
    spread: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scatter particles randomly with count proportional to concentration.
    Returns positions [N,3], colors [N,3], sizes [N].
    """
    n_species = len(concentrations)
    total_conc = concentrations.sum()
    if total_conc < 1e-15:
        fracs = np.ones(n_species) / n_species
    else:
        fracs = concentrations / total_conc

    counts = np.maximum(1, (fracs * n_particles_total).astype(int))
    total  = counts.sum()

    positions = np.random.randn(total, 3).astype(np.float32) * spread
    colors    = np.zeros((total, 3), dtype=np.float32)
    sizes     = np.zeros(total, dtype=np.float32)

    idx = 0
    for s in range(n_species):
        c = counts[s]
        color = _SPECIES_COLORS[s % len(_SPECIES_COLORS)]
        colors[idx:idx+c] = color
        sizes[idx:idx+c]  = 4.0 + 8.0 * fracs[s]
        idx += c

    return positions, colors, sizes


# ─── Main renderer class ──────────────────────────────────────────────────────

class ParticleRenderer:
    """
    Real-time OpenGL point-cloud renderer for a simulation trajectory.

    Controls:
        WASD  – pan
        Mouse drag – rotate
        Scroll – zoom
        Space – pause/resume animation
        Q/Esc – quit
    """

    def __init__(
        self,
        result: "SimulationResult",
        n_particles: int = 50_000,
        window_title: str = "ChemSim 3D Viewer",
        width: int = 1280,
        height: int = 720,
        fps_target: int = 60,
    ) -> None:
        if not OPENGL_AVAILABLE:
            raise ImportError(
                "OpenGL renderer requires: pip install chemsim[vis]\n"
                "  (pyopengl, glfw)"
            )
        self.result       = result
        self.n_particles  = n_particles
        self.title        = window_title
        self.width        = width
        self.height       = height
        self.fps_target   = fps_target
        self._running     = False
        self._paused      = False
        self._frame_idx   = 0

    def run(self, blocking: bool = True) -> None:
        """
        Open the render window.

        Parameters
        ----------
        blocking : bool
            If True (default), blocks until the window is closed.
            If False, opens the window in a background thread.
        """
        if not blocking:
            t = threading.Thread(target=self._render_loop, daemon=True)
            t.start()
            return
        self._render_loop()

    def _render_loop(self) -> None:  # pragma: no cover
        import glfw
        import OpenGL.GL as gl
        import ctypes

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)  # macOS

        window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(window)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)

        # ── Compile shaders ────────────────────────────────────────────────
        prog = _compile_program(_VERT_SHADER, _FRAG_SHADER)

        # ── Camera state ───────────────────────────────────────────────────
        cam = _Camera(self.width, self.height)
        _setup_callbacks(window, cam)

        # ── VAO / VBO ──────────────────────────────────────────────────────
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

        # ── Main loop ──────────────────────────────────────────────────────
        self._running = True
        n_frames = len(self.result.time)
        import time

        while not glfw.window_should_close(window) and self._running:
            t0 = time.perf_counter()

            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                break
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                self._paused = not self._paused

            if not self._paused:
                self._frame_idx = (self._frame_idx + 1) % n_frames

            conc = self.result.concentrations[self._frame_idx]
            pos, colors, sizes = _build_particle_array(conc, self.n_particles)

            # Pack interleaved: position (3f) + color (3f) + size (1f)
            n = len(pos)
            data = np.zeros((n, 7), dtype=np.float32)
            data[:, :3] = pos
            data[:, 3:6] = colors
            data[:, 6]   = sizes

            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
            stride = 7 * 4  # 7 floats × 4 bytes
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, False, stride, ctypes.c_void_p(24))
            gl.glEnableVertexAttribArray(2)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glUseProgram(prog)

            mvp = cam.mvp_matrix()
            loc = gl.glGetUniformLocation(prog, "MVP")
            gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, mvp)

            gl.glDrawArrays(gl.GL_POINTS, 0, n)

            # HUD overlay (FPS, time)
            t_sim  = self.result.time[self._frame_idx]
            elapsed = time.perf_counter() - t0
            fps    = 1.0 / max(elapsed, 1e-6)
            glfw.set_window_title(
                window,
                f"{self.title}  |  t={t_sim:.2f}  FPS={fps:.0f}  "
                f"[{'PAUSED' if self._paused else 'PLAYING'}]"
            )

            glfw.swap_buffers(window)

            # Frame rate cap
            sleep = 1.0 / self.fps_target - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        gl.glDeleteVertexArrays(1, [vao])
        gl.glDeleteBuffers(1, [vbo])
        gl.glDeleteProgram(prog)
        glfw.destroy_window(window)
        glfw.terminate()
        self._running = False


# ─── OpenGL helpers ───────────────────────────────────────────────────────────

def _compile_program(vert_src: str, frag_src: str) -> int:  # pragma: no cover
    import OpenGL.GL as gl

    def compile_shader(src, shader_type):
        s = gl.glCreateShader(shader_type)
        gl.glShaderSource(s, src)
        gl.glCompileShader(s)
        if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
            raise RuntimeError(f"Shader compile error: {gl.glGetShaderInfoLog(s)}")
        return s

    vs = compile_shader(vert_src, gl.GL_VERTEX_SHADER)
    fs = compile_shader(frag_src, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(f"Shader link error: {gl.glGetProgramInfoLog(prog)}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog


class _Camera:  # pragma: no cover
    """Arcball camera with pan/zoom."""

    def __init__(self, width: int, height: int) -> None:
        self.yaw    = 0.0
        self.pitch  = 20.0
        self.dist   = 20.0
        self.pan_x  = 0.0
        self.pan_y  = 0.0
        self.width  = width
        self.height = height
        self._last_mouse = None
        self._dragging   = False

    def mvp_matrix(self) -> np.ndarray:
        import math
        aspect = self.width / max(self.height, 1)

        # Perspective projection
        fov   = math.radians(45.0)
        near, far = 0.1, 500.0
        f     = 1.0 / math.tan(fov / 2)
        proj  = np.array([
            [f/aspect, 0,  0,                              0],
            [0,        f,  0,                              0],
            [0,        0,  (far+near)/(near-far),         -1],
            [0,        0,  2*far*near/(near-far),          0],
        ], dtype=np.float32)

        # View: rotate then translate back
        cy, sy = math.cos(math.radians(self.yaw)),   math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))

        rot_y = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
        rot_x = np.array([[1,0,0,0],[0,cp,-sp,0],[0,sp,cp,0],[0,0,0,1]], dtype=np.float32)
        trans = np.eye(4, dtype=np.float32)
        trans[2, 3] = -self.dist
        pan   = np.eye(4, dtype=np.float32)
        pan[0, 3]   = self.pan_x
        pan[1, 3]   = self.pan_y

        view = trans @ rot_x @ rot_y @ pan
        return proj @ view


def _setup_callbacks(window, cam: _Camera) -> None:  # pragma: no cover
    import glfw

    def mouse_button_cb(win, btn, action, mods):
        if btn == glfw.MOUSE_BUTTON_LEFT:
            cam._dragging = action == glfw.PRESS
            if cam._dragging:
                cam._last_mouse = glfw.get_cursor_pos(win)

    def cursor_pos_cb(win, xpos, ypos):
        if cam._dragging and cam._last_mouse:
            dx = xpos - cam._last_mouse[0]
            dy = ypos - cam._last_mouse[1]
            cam.yaw   += dx * 0.3
            cam.pitch  = max(-89, min(89, cam.pitch + dy * 0.3))
            cam._last_mouse = (xpos, ypos)

    def scroll_cb(win, xoff, yoff):
        cam.dist = max(1.0, cam.dist - yoff * 0.8)

    def key_cb(win, key, scan, action, mods):
        if action in (glfw.PRESS, glfw.REPEAT):
            speed = 0.3
            import math
            cy = math.cos(math.radians(cam.yaw))
            sy = math.sin(math.radians(cam.yaw))
            if key == glfw.KEY_W: cam.pan_y -= speed
            if key == glfw.KEY_S: cam.pan_y += speed
            if key == glfw.KEY_A: cam.pan_x += speed
            if key == glfw.KEY_D: cam.pan_x -= speed

    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)
    glfw.set_scroll_callback(window, scroll_cb)
    glfw.set_key_callback(window, key_cb)
