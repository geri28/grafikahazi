import taichi as ti
import time
import math
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

class AppState:
    use_gpgpu = True
    use_multithreading = True

    WIN_WIDTH = 800
    WIN_HEIGHT = 800
    UI_HEIGHT = 300

try:
    ti.init(arch=ti.gpu)
except:
    ti.init(arch=ti.cpu)

FIRE_HEIGHT = AppState.WIN_HEIGHT - AppState.UI_HEIGHT

# Taichi Field
pixels_gpu = ti.Vector.field(3, dtype=float, shape=(AppState.WIN_WIDTH, AppState.WIN_HEIGHT))

max_threads = os.cpu_count()
executor = ThreadPoolExecutor(max_workers=max_threads)

@ti.func
def fract(x): return x - ti.floor(x)
@ti.func
def mix(x, y, a): return x * (1.0 - a) + y * a
@ti.func
def hash_noise(v): return fract(ti.sin(v * 0.1) * 1000.0)

@ti.func
def snoise(uv, res):
    s = ti.Vector([1.0, 100.0, 1000.0])
    uv *= res
    uv0 = ti.floor(uv % res) * s
    uv1 = ti.floor((uv + 1.0) % res) * s
    f = fract(uv)
    f = f * f * (3.0 - 2.0 * f)
    v_x = uv0[0] + uv0[1] + uv0[2]
    v_y = uv1[0] + uv0[1] + uv0[2]
    v_z = uv0[0] + uv1[1] + uv0[2]
    v_w = uv1[0] + uv1[1] + uv0[2]
    r_x = hash_noise(v_x)
    r_y = hash_noise(v_y)
    r_z = hash_noise(v_z)
    r_w = hash_noise(v_w)
    r0 = mix(mix(r_x, r_y, f[0]), mix(r_z, r_w, f[0]), f[1])
    z_diff = uv1[2] - uv0[2]
    r_x2 = hash_noise(v_x + z_diff)
    r_y2 = hash_noise(v_y + z_diff)
    r_z2 = hash_noise(v_z + z_diff)
    r_w2 = hash_noise(v_w + z_diff)
    r1 = mix(mix(r_x2, r_y2, f[0]), mix(r_z2, r_w2, f[0]), f[1])
    return mix(r0, r1, f[2]) * 2.0 - 1.0

@ti.kernel
def render_gpgpu_kernel(t: float):
    for i, j in pixels_gpu:
        if j < AppState.UI_HEIGHT:
            pixels_gpu[i, j] = ti.Vector([0.15, 0.15, 0.15])
        else:
            fire_j = j - AppState.UI_HEIGHT
            uv_x = (i / AppState.WIN_WIDTH) - 0.5
            uv_y = (fire_j / FIRE_HEIGHT) - 0.5
            aspect = AppState.WIN_WIDTH / FIRE_HEIGHT
            uv_x *= aspect
            len_p = ti.sqrt(uv_x*uv_x + uv_y*uv_y)
            color = 3.0 - (3.0 * len_p * 2.0)
            angle = ti.atan2(uv_x, uv_y) / 6.2832 + 0.5
            coord = ti.Vector([angle, len_p * 0.4, 0.5])

            for k in range(1, 8):
                power = ti.pow(2.0, float(k))
                offset = ti.Vector([0.0, -t * 0.05, t * 0.01])
                noise_val = snoise(coord + offset, power * 16.0)
                color += (1.5 / power) * noise_val

            c_r = max(color, 0.0)
            c_g = ti.pow(max(color, 0.0), 2.0) * 0.4
            c_b = ti.pow(max(color, 0.0), 3.0) * 0.15
            pixels_gpu[i, j] = ti.Vector([c_r, c_g, c_b])

grid_x, grid_y = np.meshgrid(
    np.arange(AppState.WIN_WIDTH),
    np.arange(AppState.WIN_HEIGHT - AppState.UI_HEIGHT),
    indexing='ij'
)
simd_uv_x = (grid_x / AppState.WIN_WIDTH) - 0.5
simd_uv_y = (grid_y / FIRE_HEIGHT) - 0.5
aspect_ratio = AppState.WIN_WIDTH / FIRE_HEIGHT
simd_uv_x *= aspect_ratio
GLOBAL_LEN_P = np.sqrt(simd_uv_x**2 + simd_uv_y**2)
GLOBAL_ANGLE = np.arctan2(simd_uv_x, simd_uv_y) / 6.2832 + 0.5

def fract_np(x): return x - np.floor(x)
def mix_np(x, y, a): return x * (1.0 - a) + y * a
def hash_noise_np(v): return fract_np(np.sin(v * 0.1) * 1000.0)

def snoise_np(uv, res):
    s = np.array([1.0, 100.0, 1000.0])
    uv = uv * res
    uv0 = np.floor(uv % res)
    uv0_dot_s = uv0[0] * s[0] + uv0[1] * s[1]
    uv1 = np.floor((uv + 1.0) % res)
    uv1_dot_s = uv1[0] * s[0] + uv1[1] * s[1]
    f = fract_np(uv)
    f = f * f * (3.0 - 2.0 * f)
    z_val = 0.5 * res
    z0 = np.floor(z_val % res) * s[2]
    z1 = np.floor((z_val + 1.0) % res) * s[2]
    v_x, v_y = uv0_dot_s + z0, uv1_dot_s + z0
    v_z, v_w = uv0_dot_s + z1, uv1_dot_s + z1
    r_x, r_y = hash_noise_np(v_x), hash_noise_np(v_y)
    r_z, r_w = hash_noise_np(v_z), hash_noise_np(v_w)
    r0 = mix_np(mix_np(r_x, r_y, f[0]), mix_np(r_z, r_w, f[0]), f[1])
    return r0 * 2.0 - 1.0

def compute_fire_slice(angle_slice, len_p_slice, t):
    color = 3.0 - (3.0 * len_p_slice * 2.0)
    coord_x = angle_slice
    coord_y = len_p_slice * 0.4
    for k in range(1, 5):
        power = float(2**k)
        uv_input = np.array([coord_x, coord_y - t * 0.05])
        noise_val = snoise_np(uv_input, power * 16.0)
        color += (1.5 / power) * noise_val
    color = np.maximum(color, 0.0)
    c_r = color
    c_g = (color**2.0) * 0.4
    c_b = (color**3.0) * 0.15
    return np.dstack((c_r, c_g, c_b))

def render_cpu_single(t):
    fire_img = compute_fire_slice(GLOBAL_ANGLE, GLOBAL_LEN_P, t)
    ui_img = np.full((AppState.WIN_WIDTH, AppState.UI_HEIGHT, 3), 0.15)
    return np.concatenate((ui_img, fire_img), axis=1)

def render_cpu_multithreaded(t):
    width = AppState.WIN_WIDTH
    num_threads = max_threads
    chunk_size = width // num_threads
    futures = []
    for i in range(num_threads):
        start_x = i * chunk_size
        end_x = width if i == num_threads - 1 else (i + 1) * chunk_size
        angle_chunk = GLOBAL_ANGLE[start_x:end_x, :]
        len_p_chunk = GLOBAL_LEN_P[start_x:end_x, :]
        futures.append(executor.submit(compute_fire_slice, angle_chunk, len_p_chunk, t))
    results = [f.result() for f in futures]
    fire_img = np.concatenate(results, axis=0)
    ui_img = np.full((AppState.WIN_WIDTH, AppState.UI_HEIGHT, 3), 0.15)
    return np.concatenate((ui_img, fire_img), axis=1)

def draw_button(gui, x, y, w, h, text, is_active, mouse_x, mouse_y, is_clicked):
    color_bg = 0x444444
    if is_active:
        color_bg = 0x00AA00

    y1 = y - h/2
    y2 = y + h/2
    x1 = x
    x2 = x + w

    is_hover = (x1 < mouse_x < x2) and (y1 < mouse_y < y2)

    if is_hover and not is_active:
        color_bg = 0x666666

    gui.rect(topleft=(x1, y2), bottomright=(x2, y1), radius=2, color=color_bg)
    gui.text(text, pos=(x + 0.11, y + 0.015), font_size=20, color=0xFFFFFF)

    if is_hover and is_clicked:
        return True
    return False

def main():
    gui = ti.GUI("Fire Shader - Control Panel", res=(AppState.WIN_WIDTH, AppState.WIN_HEIGHT))

    speed_val = 1.0
    effective_time = 0.0
    last_frame_time = time.time()

    ui_top = AppState.UI_HEIGHT / AppState.WIN_HEIGHT

    slider_y = 0.30
    slider_x = 0.5
    slider_w = 0.8
    is_dragging = False

    btn_w = 0.30
    btn_h = 0.06
    btn_row_1_y = 0.20
    btn_row_2_y = 0.10

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False

        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time

        mouse_x, mouse_y = gui.get_cursor_pos()
        is_mouse_down = gui.is_pressed(ti.GUI.LMB)

        if is_mouse_down:
            if not is_dragging:
                if abs(mouse_y - slider_y) < 0.05 and abs(mouse_x - slider_x) < slider_w/2 + 0.05:
                    is_dragging = True

            if is_dragging:
                half_w = slider_w / 2
                rel_x = (mouse_x - (slider_x - half_w)) / slider_w
                speed_val = max(0.0, min(1.0, rel_x)) * 5.0
        else:
            is_dragging = False

        if gui.is_pressed(ti.GUI.RIGHT) or gui.is_pressed('Right'):
            speed_val = min(5.0, speed_val + 2.0 * dt)
        if gui.is_pressed(ti.GUI.LEFT) or gui.is_pressed('Left'):
            speed_val = max(0.0, speed_val - 2.0 * dt)

        effective_time += dt * speed_val
        t_start = time.perf_counter()

        if AppState.use_gpgpu:
            render_gpgpu_kernel(effective_time)
            ti.sync()
            gui.set_image(pixels_gpu)
        else:
            if AppState.use_multithreading:
                img = render_cpu_multithreaded(effective_time)
            else:
                img = render_cpu_single(effective_time)
            gui.set_image(img)

        t_end = time.perf_counter()
        compute_ms = (t_end - t_start) * 1000.0


        should_check_buttons = is_mouse_down and not is_dragging

        if draw_button(gui, 0.15, btn_row_1_y, btn_w, btn_h, "GPGPU", AppState.use_gpgpu, mouse_x, mouse_y, should_check_buttons):
            AppState.use_gpgpu = True

        if draw_button(gui, 0.55, btn_row_1_y, btn_w, btn_h, "CPU", not AppState.use_gpgpu, mouse_x, mouse_y, should_check_buttons):
            AppState.use_gpgpu = False

        if not AppState.use_gpgpu:
            if draw_button(gui, 0.15, btn_row_2_y, btn_w, btn_h, "1 Szal", not AppState.use_multithreading, mouse_x, mouse_y, should_check_buttons):
                AppState.use_multithreading = False

            if draw_button(gui, 0.55, btn_row_2_y, btn_w, btn_h, f"Multi ({max_threads})", AppState.use_multithreading, mouse_x, mouse_y, should_check_buttons):
                AppState.use_multithreading = True
        else:
            gui.text("(szalak kivalasztasa nem elerheto)", pos=(0.35, btn_row_2_y), color=0x555555, font_size=18)

        gui.line(begin=(0, ui_top), end=(1, ui_top), radius=1, color=0xFFFFFF)

        gui.line(begin=(slider_x - slider_w/2, slider_y), end=(slider_x + slider_w/2, slider_y), radius=4, color=0x666666)
        knob_pos_x = (slider_x - slider_w/2) + (speed_val / 5.0) * slider_w
        gui.line(begin=(slider_x - slider_w/2, slider_y), end=(knob_pos_x, slider_y), radius=4, color=0xFFaa00)
        gui.circle(pos=(knob_pos_x, slider_y), radius=12 if is_dragging else 10, color=0xFF0000 if is_dragging else 0xFFFFFF)

        gui.text(f"Sebesseg: {speed_val:.2f}x", pos=(slider_x - 0.1, slider_y + 0.05), font_size=24, color=0xFFFFFF)

        gui.text(f"Ido: {compute_ms:.2f} ms", pos=(0.70, 0.96), font_size=24, color=0xFFaa00)

        mode_text = "GPGPU" if AppState.use_gpgpu else ("CPU Multi" if AppState.use_multithreading else "CPU Single")
        gui.text(f"MOD: {mode_text}", pos=(0.02, 0.96), font_size=24, color=0x00FF00)

        gui.show()

    if executor:
        executor.shutdown()

if __name__ == "__main__":
    main()