import pygame
import numpy as np
from scipy.interpolate import interp1d
import math

class TubeMath:

    @staticmethod
    def get_tube_polygon_points(c1_data, c2_data):
        x1, y1, r1 = c1_data
        x2, y2, r2 = c2_data

        dx, dy = x2 - x1, y2 - y1
        d = np.sqrt(dx ** 2 + dy ** 2)

        base_angle = np.arctan2(dy, dx)

        try:
            val = (r1 - r2) / d
            val = max(-1.0, min(1.0, val))
            alpha = np.arccos(val)
        except ValueError:
            return None
        #4 sarokpont
        t_top = base_angle + alpha
        t1x_top = x1 + r1 * np.cos(t_top)
        t1y_top = y1 + r1 * np.sin(t_top)
        t2x_top = x2 + r2 * np.cos(t_top)
        t2y_top = y2 + r2 * np.sin(t_top)

        t_bot = base_angle - alpha
        t1x_bot = x1 + r1 * np.cos(t_bot)
        t1y_bot = y1 + r1 * np.sin(t_bot)
        t2x_bot = x2 + r2 * np.cos(t_bot)
        t2y_bot = y2 + r2 * np.sin(t_bot)

        return np.array([
            [t1x_top, t1y_top],
            [t2x_top, t2y_top],
            [t2x_bot, t2y_bot],
            [t1x_bot, t1y_bot]
        ])

    @staticmethod
    def interpolate_circles(circles_data, level):
        n = len(circles_data)
        if n < 2 or level == 0:
            return circles_data

        x = np.array([c[0] for c in circles_data])
        y = np.array([c[1] for c in circles_data])
        r = np.array([c[2] for c in circles_data])

        t = np.arange(n)

        kind_pos = 'linear'
        if n == 3:
            kind_pos = 'quadratic'
        elif n >= 4:
            kind_pos = 'cubic'

        # SUGÁR interpolációja
        kind_radius = 'linear'

        fx = interp1d(t, x, kind=kind_pos)
        fy = interp1d(t, y, kind=kind_pos)
        fr = interp1d(t, r, kind=kind_radius)

        points_per_segment = int(level * 5)
        total_points = (n - 1) * points_per_segment + 1

        t_new = np.linspace(0, n - 1, total_points)

        x_new = fx(t_new)
        y_new = fy(t_new)
        r_new = fr(t_new)

        return list(zip(x_new, y_new, r_new))

def main():
    pygame.init()

    WIDTH, HEIGHT = 1000, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("körös feladat")
    clock = pygame.time.Clock()

    BG_COLOR = (20, 20, 20)
    TUBE_COLOR = (0, 200, 255)
    KEYFRAME_COLOR = (255, 255, 0)
    GHOST_COLOR = (100, 100, 100)
    TEXT_COLOR = (255, 255, 255)

    circles = []
    temp_center = None

    INTERPOLATION_LEVEL = 5

    number_font = pygame.font.SysFont("Arial", 20, bold=True)
    ui_font = pygame.font.SysFont("Helvetica", 18)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if temp_center is None:
                        temp_center = mouse_pos
                    else:
                        dx = mouse_pos[0] - temp_center[0]
                        dy = mouse_pos[1] - temp_center[1]
                        radius = math.sqrt(dx**2 + dy**2)
                        radius = max(5.0, radius)

                        circles.append((temp_center[0], temp_center[1], radius))
                        temp_center = None

                elif event.button == 3:
                    if temp_center is not None:
                        temp_center = None
                    elif circles:
                        circles.pop()

        screen.fill(BG_COLOR)

        if len(circles) >= 2:
            render_circles = TubeMath.interpolate_circles(circles, INTERPOLATION_LEVEL)
        else:
            render_circles = circles

        if len(render_circles) >= 2:
            for i in range(len(render_circles) - 1):
                c1 = render_circles[i]
                c2 = render_circles[i+1]
                poly_points = TubeMath.get_tube_polygon_points(c1, c2)

                if poly_points is not None:
                    pygame.draw.polygon(screen, TUBE_COLOR, poly_points.tolist())

            for x, y, r in render_circles:
                pygame.draw.circle(screen, TUBE_COLOR, (int(x), int(y)), int(r))

        for i, (x, y, r) in enumerate(circles):
            pygame.draw.circle(screen, KEYFRAME_COLOR, (int(x), int(y)), int(r), 2)

            text_surf = number_font.render(str(i + 1), True, TEXT_COLOR)

            text_rect = text_surf.get_rect(center=(int(x), int(y)))

            screen.blit(text_surf, text_rect)

        if temp_center is not None:
            dx = mouse_pos[0] - temp_center[0]
            dy = mouse_pos[1] - temp_center[1]
            current_radius = math.sqrt(dx**2 + dy**2)
            pygame.draw.circle(screen, GHOST_COLOR, temp_center, int(current_radius), 1)
            pygame.draw.line(screen, GHOST_COLOR, temp_center, mouse_pos)

        texts = [
            f"Körök száma: {len(circles)}",
            "1. bal klikk: kör",
            "2. bal klikk: sugár",
            "Jobb klikk: törlés",
            f"interpoláció: {INTERPOLATION_LEVEL}"
        ]
        y = 10
        for line in texts:
            text_surface = ui_font.render(line, True, (200, 200, 200))
            screen.blit(text_surface, (10, y))
            y += text_surface.get_height() + 5
        #text = ui_font.render(f"Körök száma: {len(circles)} | 1. bal klikk: kör | 2. bal klikk: sugár | Jobb klikk: törlés | interpoláció: {INTERPOLATION_LEVEL}", True, (200, 200, 200))
        #screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()