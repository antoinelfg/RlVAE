from manim import *
from manim_slides import Slide
import numpy as np
import random
import os
from pathlib import Path
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm

# ---------- GLOBAL STYLE ----------
BACKGROUND      = "#0d1117"
ENCODER_COLOR   = BLUE_B
DECODER_COLOR   = GREEN_C
MANIFOLD_COLOR  = GREEN_A
METRIC_COLOR    = YELLOW_D
FORWARD_COLOR   = BLUE_C
REVERSE_COLOR   = GREEN_C
NOISE_COLOR     = RED_D
EQUATION_COLOR  = WHITE
TEXT_COLOR      = "#e6edf3"
LEGEND_COLOR    = "#2d333b"

# ---------- VIDEO CONFIG ----------
config.background_color = ManimColor.from_hex(BACKGROUND)
config.frame_rate = 60
config.pixel_height = 1080
config.pixel_width = 1920
config.quality = "high_quality"
config.media_dir = "media"
config.video_dir = "gugus_visualization/1080p60"
config.disable_caching = True

# Manim Slides specific configuration
config.slides_dir = "slides/files"
config.slide_format = "mp4"
config.slide_quality = "high_quality"
config.slide_resolution = (1920, 1080)
config.slide_fps = 60

def create_directories():
    base_dir = Path(__file__).parent
    resolution = f"{config.pixel_height}p{config.frame_rate}"
    
    # Create all necessary directories
    dirs = [
        base_dir / "media/videos/gugus_visualization",
        base_dir / "media/videos/gugus_visualization/1080p60",
        base_dir / "media/videos/gugus_visualization/1080p60/partial_movie_files",
        base_dir / "media/videos/gugus_visualization/1080p60/partial_movie_files/GUGUSVisualization",
        base_dir / "slides/files/GUGUSVisualization",
        base_dir / "media/Tex",
        base_dir / "media/texts",
        base_dir / "media/images",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set up video paths
    video_path = base_dir / "media/videos/gugus_visualization/1080p60/GUGUSVisualization.mp4"
    slides_video_path = base_dir / "slides/files/GUGUSVisualization/GUGUSVisualization.mp4"
    
    # Create symlink if needed
    try:
        if video_path.exists() and not slides_video_path.exists():
            os.symlink(video_path, slides_video_path)
        elif slides_video_path.exists() and not video_path.exists():
            os.symlink(slides_video_path, video_path)
    except FileExistsError:
        pass

def _sqrt_bar_alpha(t):        # toy schedule, monotone in t ∈ [0,1]
    return np.exp(-2*t)        # fast decay → clearly visible noise

# ------------------------------------------------------------------ #
class GUGUSVisualization(Slide):

    # ---------- boiler-plate ----------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = ManimColor.from_hex(BACKGROUND)
        random.seed(1)  # reproducible forward / reverse noise
        create_directories()  # Ensure all directories exist

    # =============================================================== #
    #                         CONSTRUCT                              #
    # =============================================================== #
    def construct(self):
        self.intro()
        self.encoder_step()
        self.manifold_metric()
        self.riemannian_sampling()
        self.forward_diffusion()
        self.reverse_diffusion()
        self.decoding()

    # =============================================================== #
    #                           INTRO                                #
    # =============================================================== #
    def _create_legend(self, items, position=RIGHT):
        """Create a legend box with items"""
        legend_box = Rectangle(
            height=len(items) * 0.5 + 0.5,
            width=2.5,
            color=LEGEND_COLOR,
            fill_opacity=0.3,
            stroke_width=1
        )
        legend_box.to_edge(position)
        
        legend_items = VGroup()
        for i, (color, text) in enumerate(items):
            dot = Dot(color=color)
            label = Text(text, color=TEXT_COLOR, font_size=24)
            group = VGroup(dot, label).arrange(RIGHT, buff=0.2)
            group.next_to(legend_box.get_top(), DOWN, buff=0.2)
            if i > 0:
                group.next_to(legend_items[i-1], DOWN, buff=0.2)
            legend_items.add(group)
        
        return VGroup(legend_box, legend_items)

    def intro(self):
        # Title with gradient
        title = Text("GUGUS – Generative Uniform-Geodesic Unsupervised System",
                     gradient=(ORANGE, RED), weight=BOLD).scale(0.65)
        title.to_edge(UP, buff=0.5)

        # Detailed mathematical description with animations
        equations = VGroup(
            MathTex(r"\text{Encoder: } q_\phi(z|x) = \mathcal{N}(z|\mu_\phi(x), \Sigma_\phi(x))", 
                   color=EQUATION_COLOR),
            MathTex(r"\text{Metric: } G(z) = \sum_{k=1}^K w_k(z) M_k", 
                   color=EQUATION_COLOR),
            MathTex(r"\text{Sampling: } p(z) \propto \sqrt{\det G(z)}", 
                   color=EQUATION_COLOR),
            MathTex(r"\text{Diffusion: } q(z_t|z_{t-1}) = \mathcal{N}(z_t|\sqrt{\alpha_t}z_{t-1}, (1-\alpha_t)I)", 
                   color=EQUATION_COLOR),
            MathTex(r"\text{Decoder: } p_\theta(x|z) = \mathcal{N}(x|\mu_\theta(z), \Sigma_\theta(z))", 
                   color=EQUATION_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.5)
        equations.next_to(title, DOWN, buff=0.6)

        # Create legend
        legend_items = [
            (ENCODER_COLOR, "Encoder"),
            (DECODER_COLOR, "Decoder"),
            (MANIFOLD_COLOR, "Manifold"),
            (METRIC_COLOR, "Metric"),
            (FORWARD_COLOR, "Forward"),
            (REVERSE_COLOR, "Reverse"),
            (NOISE_COLOR, "Noise")
        ]
        legend = self._create_legend(legend_items)

        # Animated explanations
        explanations = VGroup(
            Text("1. Encoder maps input to latent space", color=TEXT_COLOR),
            Text("2. Riemannian metric defines local geometry", color=TEXT_COLOR),
            Text("3. Uniform sampling under metric", color=TEXT_COLOR),
            Text("4. Forward/Reverse diffusion process", color=TEXT_COLOR),
            Text("5. Decoder reconstructs from latent space", color=TEXT_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).scale(0.4)
        explanations.next_to(equations, DOWN, buff=0.4)

        # Animate title
        self.play(FadeIn(title, shift=DOWN))
        self.pause()

        # Animate equations and explanations together
        for eq, exp in zip(equations, explanations):
            self.play(
                Write(eq),
                Write(exp),
                run_time=1.5
            )
            self.pause()

        # Show legend
        self.play(FadeIn(legend))
        self.pause()

        # Fade out everything
        self.play(FadeOut(VGroup(title, equations, explanations, legend)))
        self.pause()

    # =============================================================== #
    #                     ENCODER  STEP                               #
    # =============================================================== #
    def encoder_step(self):
        # Create neural network architecture
        title = Text("Encoder Architecture", color=ENCODER_COLOR).to_edge(UP)
        
        # Input layer with animated data points
        input_layer = Rectangle(height=2, width=2, color=YELLOW)
        input_layer.shift(LEFT*4)
        input_label = MathTex("x", color=YELLOW).next_to(input_layer, UP)
        
        # Create animated data points
        data_points = VGroup()
        for _ in range(10):
            point = Dot(color=YELLOW, radius=0.05)
            point.move_to(input_layer.get_center() + np.random.uniform(-0.8, 0.8, 3))
            data_points.add(point)
        
        # Hidden layers with animated connections
        hidden_layers = VGroup()
        for i in range(3):
            layer = Rectangle(height=1.5, width=1, color=ENCODER_COLOR)
            layer.shift(LEFT*(1-i))
            hidden_layers.add(layer)
        
        # Output layer with animated latent points
        output_layer = Rectangle(height=2, width=2, color=NOISE_COLOR)
        output_layer.shift(RIGHT*4)
        output_label = MathTex("z_T", color=NOISE_COLOR).next_to(output_layer, UP)
        
        # Create animated connections
        connections = VGroup()
        for i in range(4):
            if i == 0:
                start = input_layer.get_right()
                end = hidden_layers[0].get_left()
            elif i == 3:
                start = hidden_layers[2].get_right()
                end = output_layer.get_left()
            else:
                start = hidden_layers[i-1].get_right()
                end = hidden_layers[i].get_left()
            connections.add(Arrow(start, end, buff=0.2))
        
        # Mathematical details with animations
        math_details = VGroup(
            MathTex(r"\mu_\phi(x) = W_2\sigma(W_1x + b_1) + b_2", color=EQUATION_COLOR),
            MathTex(r"\Sigma_\phi(x) = \text{diag}(\exp(W_4\sigma(W_3x + b_3) + b_4))", color=EQUATION_COLOR)
        ).arrange(DOWN, buff=0.3).scale(0.6)
        math_details.next_to(output_layer, DOWN, buff=0.5)
        
        # Animate the architecture
        self.play(FadeIn(title))
        self.pause()
        
        # Animate input layer and data points
        self.play(
            Create(input_layer),
            Write(input_label),
            FadeIn(data_points, lag_ratio=0.1)
        )
        self.pause()
        
        # Animate hidden layers and connections
        self.play(
            Create(hidden_layers),
            *[GrowArrow(conn) for conn in connections]
        )
        self.pause()
        
        # Animate output layer and mathematical details
        self.play(
            Create(output_layer),
            Write(output_label),
            Write(math_details)
        )
        self.pause()
        
        # Animate data transformation
        self.play(
            data_points.animate.move_to(output_layer.get_center()),
            run_time=2
        )
        self.pause()
        
        # Fade out everything
        self.play(FadeOut(VGroup(title, input_layer, hidden_layers, output_layer, 
                                input_label, output_label, connections, math_details, data_points)))
        self.pause()

    # =============================================================== #
    #                 MANIFOLD  +  METRIC                             #
    # =============================================================== #
    def _make_axes(self) -> Axes:
        return Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            axis_config={"stroke_color": GREY_B, "stroke_width": 1},
        ).scale(0.8)

    def _make_manifold(self, axes: Axes) -> VMobject:
        """Create a more accurate manifold representation with annotations"""
        # Define the manifold equation
        manifold_eq = MathTex(
            r"\mathcal{M} = \{z \in \mathbb{R}^d | f(z) = 0\}",
            color=MANIFOLD_COLOR
        ).scale(0.6).to_edge(UP).shift(DOWN)
        
        # Create the manifold curve
        t = np.linspace(0, 2*np.pi, 200)
        pts = [
            axes.c2p((1.2 + 0.25*np.cos(3*u))*np.cos(u),
                     (1.2 + 0.25*np.cos(3*u))*np.sin(u),
                     0)
            for u in t
        ]
        curve = VMobject(color=MANIFOLD_COLOR, stroke_width=4)
        curve.set_points_smoothly(pts)
        
        # Add tangent space annotations
        tangent_spaces = VGroup()
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            point = [1.2*np.cos(angle), 1.2*np.sin(angle)]
            tangent = Line(
                axes.c2p(*point, 0),
                axes.c2p(*(point + [np.cos(angle), np.sin(angle)]), 0),
                color=MANIFOLD_COLOR,
                stroke_width=2
            )
            tangent_spaces.add(tangent)
        
        return VGroup(manifold_eq, curve, tangent_spaces)

    def _metric_ellipsoid(self, axes, center, angle=0, scale=0.3):
        """Create a more accurate metric tensor visualization with annotations"""
        # Define the metric tensor equation
        metric_eq = MathTex(
            r"G(z) = \begin{pmatrix} g_{11} & g_{12} \\ g_{21} & g_{22} \end{pmatrix}",
            color=METRIC_COLOR
        ).scale(0.6)
        
        # Position the equation above the center point
        center_3d = axes.c2p(center[0], center[1], 0)
        metric_eq.next_to(center_3d, UP, buff=0.2)
        
        # Create the metric tensor visualization
        theta = angle
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        D = np.diag([1.0, 0.5])
        G = R @ D @ R.T
        
        # Create ellipse points
        t = np.linspace(0, 2*np.pi, 100)
        points = np.array([np.cos(t), np.sin(t)])
        transformed = G @ points
        
        # Scale and translate
        transformed = scale * transformed
        transformed[0] += center[0]
        transformed[1] += center[1]
        
        # Convert to Manim points
        points = [axes.c2p(x, y, 0) for x, y in zip(transformed[0], transformed[1])]
        metric = VMobject(color=METRIC_COLOR, fill_opacity=0.2, stroke_width=2)
        metric.set_points_smoothly(points)
        
        # Add metric tensor components as annotations
        components = VGroup(
            MathTex(r"g_{11} = 1.0", color=METRIC_COLOR).scale(0.4),
            MathTex(r"g_{12} = g_{21} = 0.0", color=METRIC_COLOR).scale(0.4),
            MathTex(r"g_{22} = 0.5", color=METRIC_COLOR).scale(0.4)
        ).arrange(DOWN, buff=0.1)
        components.next_to(metric_eq, RIGHT, buff=0.2)
        
        return VGroup(metric_eq, metric, components)

    def manifold_metric(self):
        title = Text("Riemannian Manifold & Metric Tensor", color=MANIFOLD_COLOR).to_edge(UP)
        
        # Create axes
        axes = self._make_axes()
        
        # Create manifold with annotations
        manifold_group = self._make_manifold(axes)
        manifold_eq, manifold, tangent_spaces = manifold_group
        
        # Create metric tensors with annotations
        metric_tensors = VGroup()
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            center = [1.2*np.cos(angle), 1.2*np.sin(angle)]
            metric_group = self._metric_ellipsoid(axes, center, angle=angle)
            metric_tensors.add(metric_group)
        
        # Mathematical details with animations
        math_details = VGroup(
            MathTex(r"G(z) = \sum_{k=1}^K w_k(z) M_k", color=EQUATION_COLOR),
            MathTex(r"w_k(z) = \frac{\exp(-\frac{\|z-c_k\|^2}{2\lambda})}{\sum_{j=1}^K \exp(-\frac{\|z-c_j\|^2}{2\lambda})}", 
                   color=EQUATION_COLOR),
            MathTex(r"\text{Volume element: } \sqrt{\det G(z)}", color=EQUATION_COLOR)
        ).arrange(DOWN, buff=0.3).scale(0.6)
        math_details.to_edge(RIGHT)
        
        # Create legend for metric visualization
        metric_legend = self._create_legend([
            (METRIC_COLOR, "Metric Tensor"),
            (MANIFOLD_COLOR, "Tangent Space"),
            (EQUATION_COLOR, "Components")
        ], position=LEFT)
        
        # Animate the visualization
        self.play(FadeIn(title))
        self.pause()
        
        self.play(Create(axes))
        self.pause()
        
        self.play(Write(manifold_eq))
        self.pause()
        
        self.play(Create(manifold))
        self.pause()
        
        self.play(Create(tangent_spaces))
        self.pause()
        
        self.play(FadeIn(metric_legend))
        self.pause()
        
        # Animate metric tensors one by one
        for metric_group in metric_tensors:
            self.play(FadeIn(metric_group))
            self.pause()
        
        self.play(Write(math_details))
        self.pause()
        
        # Animate metric tensor transformations
        for metric_group in metric_tensors:
            metric_eq, metric, components = metric_group
            self.play(
                metric.animate.scale(1.2),
                run_time=0.5
            )
            self.play(
                metric.animate.scale(1/1.2),
                run_time=0.5
            )
        
        self.play(FadeOut(VGroup(title, axes, manifold_group, metric_tensors, math_details, metric_legend)))
        self.pause()

    # =============================================================== #
    #                   RIEMANNIAN SAMPLING                           #
    # =============================================================== #
    def riemannian_sampling(self):
        title = Text("Riemannian Uniform Sampling", color=YELLOW).to_edge(UP)
        
        # Create axes and manifold
        axes = self._make_axes()
        manifold_group = self._make_manifold(axes)
        manifold_eq, manifold, tangent_spaces = manifold_group
        
        # Sampling equation with animation
        sampling_eq = MathTex(
            r"p(z) = \frac{\sqrt{\det G(z)}}{\int \sqrt{\det G(z)} dz}",
            color=EQUATION_COLOR
        ).scale(0.7).to_edge(UP).shift(DOWN)
        
        # Create sampling points with animations
        accepted = VGroup()
        rejected = VGroup()
        for _ in range(50):
            theta = random.random() * 2 * np.pi
            r = 1.7 * random.random()
            x, y = r * np.cos(theta), r * np.sin(theta)
            point = axes.c2p(x, y, 0)
            
            # More accurate acceptance probability
            G = np.array([[1.0, 0.0], [0.0, 0.5]])  # Simplified metric
            det_G = np.linalg.det(G)
            acceptance_prob = np.sqrt(det_G) / 2.0  # Simplified normalization
            
            if random.random() < acceptance_prob:
                dot = Dot(point, color=YELLOW, radius=0.05)
                accepted.add(dot)
            else:
                dot = Dot(point, color=RED_B, radius=0.05)
                rejected.add(dot)
        
        # Create legend for sampling
        sampling_legend = self._create_legend([
            (YELLOW, "Accepted"),
            (RED_B, "Rejected")
        ], position=LEFT)
        
        # Animate the visualization
        self.play(FadeIn(title))
        self.pause()
        
        self.play(Create(axes), Create(manifold))
        self.pause()
        
        self.play(Write(sampling_eq))
        self.pause()
        
        self.play(FadeIn(sampling_legend))
        self.pause()
        
        # Animate sampling points
        self.play(
            FadeIn(accepted, lag_ratio=0.1),
            FadeIn(rejected, lag_ratio=0.1)
        )
        self.pause()
        
        # Animate accepted points moving to manifold
        self.play(
            accepted.animate.move_to(manifold.get_center()),
            run_time=2
        )
        self.pause()
        
        self.play(FadeOut(VGroup(title, axes, manifold_group, sampling_eq, accepted, rejected, sampling_legend)))
        self.pause()

    # =============================================================== #
    #                  FORWARD  DIFFUSION                             #
    # =============================================================== #
    def forward_diffusion(self):
        title = Text("Forward Diffusion Process", color=FORWARD_COLOR).to_edge(UP)
        
        # Create axes
        axes = self._make_axes()
        
        # Diffusion equation with animation
        diffusion_eq = MathTex(
            r"q(z_t|z_{t-1}) = \mathcal{N}(z_t|\sqrt{\alpha_t}z_{t-1}, (1-\alpha_t)I)",
            color=EQUATION_COLOR
        ).scale(0.7).to_edge(UP).shift(DOWN)
        
        # Create diffusion path with animations
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 1.0])
        path = self._make_manifold(axes)
        
        # Add noise points with animations
        noise_points = VGroup()
        for t in np.linspace(0, 1, 10):
            alpha = _sqrt_bar_alpha(t)
            point = np.sqrt(alpha) * start + np.sqrt(1-alpha) * end
            noise = np.random.normal(0, 0.1, 2)
            point += noise
            dot = Dot(axes.c2p(*point, 0), color=NOISE_COLOR, radius=0.05)
            noise_points.add(dot)
        
        # Create legend for diffusion
        diffusion_legend = self._create_legend([
            (FORWARD_COLOR, "Forward Path"),
            (NOISE_COLOR, "Noise Points")
        ], position=LEFT)
        
        # Animate the visualization
        self.play(FadeIn(title))
        self.pause()
        
        self.play(Create(axes))
        self.pause()
        
        self.play(Write(diffusion_eq))
        self.pause()
        
        self.play(FadeIn(diffusion_legend))
        self.pause()
        
        self.play(Create(path))
        self.pause()
        
        # Animate noise points appearing along the path
        for point in noise_points:
            self.play(FadeIn(point))
            self.pause()
        
        # Animate noise points moving along the path
        self.play(
            noise_points.animate.move_to(path.get_center()),
            run_time=2
        )
        self.pause()
        
        self.play(FadeOut(VGroup(title, axes, diffusion_eq, path, noise_points, diffusion_legend)))
        self.pause()

    # =============================================================== #
    #                  REVERSE  DIFFUSION                             #
    # =============================================================== #
    def reverse_diffusion(self):
        title = Text("Reverse Diffusion Process", color=REVERSE_COLOR).to_edge(UP)
        
        # Create axes
        axes = self._make_axes()
        
        # Reverse diffusion equation with animation
        reverse_eq = MathTex(
            r"p_\theta(z_{t-1}|z_t) = \mathcal{N}(z_{t-1}|\mu_\theta(z_t,t), \Sigma_\theta(z_t,t))",
            color=EQUATION_COLOR
        ).scale(0.7).to_edge(UP).shift(DOWN)
        
        # Create reverse path with animations
        start = np.array([1.0, 1.0])
        end = np.array([0.0, 0.0])
        path = self._make_manifold(axes)
        
        # Add denoising points with animations
        denoising_points = VGroup()
        for t in np.linspace(0, 1, 10):
            alpha = _sqrt_bar_alpha(1-t)
            point = np.sqrt(alpha) * start + np.sqrt(1-alpha) * end
            noise = np.random.normal(0, 0.1, 2)
            point += noise
            dot = Dot(axes.c2p(*point, 0), color=REVERSE_COLOR, radius=0.05)
            denoising_points.add(dot)
        
        # Create legend for reverse diffusion
        reverse_legend = self._create_legend([
            (REVERSE_COLOR, "Reverse Path"),
            (NOISE_COLOR, "Denoising Points")
        ], position=LEFT)
        
        # Animate the visualization
        self.play(FadeIn(title))
        self.pause()
        
        self.play(Create(axes))
        self.pause()
        
        self.play(Write(reverse_eq))
        self.pause()
        
        self.play(FadeIn(reverse_legend))
        self.pause()
        
        self.play(Create(path))
        self.pause()
        
        # Animate denoising points appearing along the path
        for point in denoising_points:
            self.play(FadeIn(point))
            self.pause()
        
        # Animate denoising points moving along the path
        self.play(
            denoising_points.animate.move_to(path.get_center()),
            run_time=2
        )
        self.pause()
        
        self.play(FadeOut(VGroup(title, axes, reverse_eq, path, denoising_points, reverse_legend)))
        self.pause()

    # =============================================================== #
    #                    DECODING                                    #
    # =============================================================== #
    def decoding(self):
        title = Text("Decoder Architecture", color=DECODER_COLOR).to_edge(UP)
        
        # Input layer with animated latent points
        input_layer = Rectangle(height=2, width=2, color=FORWARD_COLOR)
        input_layer.shift(LEFT*4)
        input_label = MathTex("z_T", color=FORWARD_COLOR).next_to(input_layer, UP)
        
        # Create animated latent points
        latent_points = VGroup()
        for _ in range(10):
            point = Dot(color=FORWARD_COLOR, radius=0.05)
            point.move_to(input_layer.get_center() + np.random.uniform(-0.8, 0.8, 3))
            latent_points.add(point)
        
        # Hidden layers with animated connections
        hidden_layers = VGroup()
        for i in range(3):
            layer = Rectangle(height=1.5, width=1, color=DECODER_COLOR)
            layer.shift(RIGHT*(1-i))
            hidden_layers.add(layer)
        
        # Output layer with animated reconstructed points
        output_layer = Rectangle(height=2, width=2, color=YELLOW_C)
        output_layer.shift(RIGHT*4)
        output_label = MathTex(r"\hat{x}", color=YELLOW).next_to(output_layer, UP)
        
        # Create animated connections
        connections = VGroup()
        for i in range(4):
            if i == 0:
                start = input_layer.get_right()
                end = hidden_layers[0].get_left()
            elif i == 3:
                start = hidden_layers[2].get_right()
                end = output_layer.get_left()
            else:
                start = hidden_layers[i-1].get_right()
                end = hidden_layers[i].get_left()
            connections.add(Arrow(start, end, buff=0.2))
        
        # Mathematical details with animations
        math_details = VGroup(
            MathTex(r"\mu_\theta(z) = W_2\sigma(W_1z + b_1) + b_2", color=EQUATION_COLOR),
            MathTex(r"\Sigma_\theta(z) = \text{diag}(\exp(W_4\sigma(W_3z + b_3) + b_4))", color=EQUATION_COLOR),
            MathTex(r"p_\theta(x|z) = \mathcal{N}(x|\mu_\theta(z), \Sigma_\theta(z))", color=EQUATION_COLOR)
        ).arrange(DOWN, buff=0.3).scale(0.6)
        math_details.next_to(output_layer, DOWN, buff=0.5)
        
        # Create legend for decoder
        decoder_legend = self._create_legend([
            (FORWARD_COLOR, "Latent Points"),
            (DECODER_COLOR, "Hidden Layers"),
            (YELLOW_C, "Reconstructed Points")
        ], position=LEFT)
        
        # Animate the architecture
        self.play(FadeIn(title))
        self.pause()
        
        self.play(FadeIn(decoder_legend))
        self.pause()
        
        # Animate input layer and latent points
        self.play(
            Create(input_layer),
            Write(input_label),
            FadeIn(latent_points, lag_ratio=0.1)
        )
        self.pause()
        
        # Animate hidden layers and connections
        self.play(
            Create(hidden_layers),
            *[GrowArrow(conn) for conn in connections]
        )
        self.pause()
        
        # Animate output layer and mathematical details
        self.play(
            Create(output_layer),
            Write(output_label),
            Write(math_details)
        )
        self.pause()
        
        # Animate latent points transforming to reconstructed points
        self.play(
            latent_points.animate.move_to(output_layer.get_center()),
            run_time=2
        )
        self.pause()
        
        # Fade out everything
        self.play(FadeOut(VGroup(title, input_layer, hidden_layers, output_layer, 
                                input_label, output_label, connections, math_details, 
                                latent_points, decoder_legend)))
        self.pause()