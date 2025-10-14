import tkinter as tk
import numpy as np

class PointPlacerApp:
    def __init__(self, root):
        self.gui(root)

    def gui(self, root):
        self.root = root

        self.canvas = tk.Canvas(root, width=1000, height=800, bg="white")
        self.canvas.pack(pady=10, padx=10)

        self.canvas.bind("<Button-1>", self.placed_points)
        self.canvas.bind("<B1-Motion>", self.move_point)
        self.canvas.bind("<ButtonRelease-1>", self.release_point)

        self.weight_slider = tk.Scale(
            root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            label="Súly", length=400,
            command=self.slider
        )
        self.weight_slider.set(0.0)
        self.weight_slider.pack(pady=(10, 0))

        iter_frame = tk.Frame(root)
        iter_frame.pack(pady=5)
        tk.Label(iter_frame, text=" iterációk száma: ").pack(side="left")
        self.iter_var = tk.StringVar(value="0")
        self.iter_spinbox = tk.Spinbox(
            iter_frame, from_=0, to=6, width=3, textvariable=self.iter_var,
            command=self.iteration, state="readonly"
        )
        self.iter_spinbox.pack(side="left")

        self.approx_var = tk.BooleanVar(value=False)
        self.approx_check = tk.Checkbutton(
            root, text="Approximáció",
            variable=self.approx_var, command=self.checkbox
        )
        self.approx_check.pack(pady=5)

        self.points = []
        self.point_items = []
        self.selected_point_index = None

    def placed_points(self, event):
        x, y = event.x, event.y
        radius = 5
        for i, (px, py) in enumerate(self.points):
            if (x - px)**2 + (y - py)**2 < (radius + 3)**2:
                self.selected_point_index = i
                return
        point_id = self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill="black", outline="black"
        )
        self.points.append((x, y))
        self.point_items.append(point_id)
        self.selected_point_index = None
        self.redraw()

    def move_point(self, event):
        if self.selected_point_index is not None:
            x, y = event.x, event.y
            self.points[self.selected_point_index] = (x, y)
            radius = 5
            item_id = self.point_items[self.selected_point_index]
            self.canvas.coords(item_id, x - radius, y - radius, x + radius, y + radius)
            self.redraw()

    def release_point(self, event):
        self.selected_point_index = None

    def slider(self, value=None):
        self.redraw()

    def iteration(self):
        self.redraw()

    def checkbox(self):
        self.redraw()

    def redraw(self):
        self.canvas.delete("subdiv", "control_poly")

        if len(self.points) < 3:
            return

        try:
            iterations = int(self.iter_var.get())
            slider_val = float(self.weight_slider.get())
            is_approximating = self.approx_var.get()
        except (ValueError, tk.TclError):
            return

        control_poly_points = self.points + [self.points[0]]
        flat_control_points = [c for p in control_poly_points for c in p]
        self.canvas.create_line(
            flat_control_points, fill="gray", dash=(2, 4), width=1, tags="control_poly"
        )
        
        min_points = 3 if is_approximating else 4
        if len(self.points) < min_points:
            return

        subdivided_points = self.apply_subdivision(
            self.points, iterations, is_approximating, slider_val
        )

        if subdivided_points:
            drawable_points = subdivided_points + [subdivided_points[0]]
            flat_points = [c for p in drawable_points for c in p]
            self.canvas.create_line(
                flat_points, fill="blue", width=2, tags="subdiv"
            )

    def apply_subdivision(self, points, iterations, is_approximating, slider_val):
        if is_approximating:
            return self.approximation_subdivision(points, iterations, slider_val)
        else:
            weight = slider_val * 0.25
            result = list(points)
            for _ in range(iterations):
                result = self.interpolating_subdivision(result, weight)
            return result

    def approximation_subdivision(self, points, iterations, slider_value):
        if len(points) < 3:
            return points
            
        closed_points = points + [points[0]]
        new_points = np.array(closed_points)
        
        alpha = 0.1 + slider_value * 0.3

        for _ in range(iterations):
            n = len(new_points) - 1 
            if n < 2: 
                break

            subdivided_points = []
            
            for i in range(n):
                P0 = new_points[i]
                P1 = new_points[i + 1]
                
                Q = (1.0 - alpha) * P0 + alpha * P1
                R = alpha * P0 + (1.0 - alpha) * P1

                subdivided_points.append(Q)
                subdivided_points.append(R)
            
            if subdivided_points:
                new_points = np.vstack(subdivided_points)
                new_points = np.vstack([new_points, new_points[0]])

        result = new_points.tolist()
        if len(result) > 1 and result[0] == result[-1]:
            result = result[:-1]  
            
        return result

    def interpolating_subdivision(self, pts, w):
        new_pts = []
        n = len(pts)
        for i in range(n):
            new_pts.append(pts[i]) 
            
            p_im1 = pts[(i - 1 + n) % n]
            p_i = pts[i]
            p_ip1 = pts[(i + 1) % n]
            p_ip2 = pts[(i + 2) % n]
            
            mid_x = 0.5 * (p_i[0] + p_ip1[0])
            mid_y = 0.5 * (p_i[1] + p_ip1[1])
            
            offset_x = w * ((p_i[0] - p_im1[0]) - (p_ip2[0] - p_ip1[0]))
            offset_y = w * ((p_i[1] - p_im1[1]) - (p_ip2[1] - p_ip1[1]))
            
            new_pts.append((mid_x + offset_x, mid_y + offset_y))
        return new_pts

if __name__ == "__main__":
    root = tk.Tk()
    app = PointPlacerApp(root)
    root.mainloop()