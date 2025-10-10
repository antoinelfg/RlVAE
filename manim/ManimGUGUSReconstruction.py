from manim import *
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from colour import Color

# Configuration settings
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.media_width = "100%"
config.background_color = BLACK
config.frame_height = 8.0
config.frame_width = 14.222222222222221
config.output_file = "GUGUSModelVisualization"
config.format = "mp4"
config.disable_caching = True
config.flush_cache = True

# Custom color scheme
COLORS = {
    "patient": "#4CAF50",  # Green
    "latent": "#9C27B0",   # Purple
    "encoder": "#2196F3",  # Blue
    "decoder": "#FF9800",  # Orange
    "diffusion": "#E91E63", # Pink
    "highlight": "#FFC107", # Amber
    "background": "#212121", # Dark Gray
    "text": "#FFFFFF",     # White
    "arrow": "#90CAF9",    # Light Blue
    "grid": "#424242",     # Gray
}

class BlockScheme(VGroup):
    """A class to create and manage the block scheme overview."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = {}
        self.arrows = {}
        self.create_blocks()
        self.create_connections()
        
    def create_blocks(self):
        # Create main blocks
        blocks_info = {
            "data": ("Patient\nTrajectories", COLORS["patient"]),
            "encoder": ("Encoder\nNetwork", COLORS["encoder"]),
            "latent": ("Latent\nSpace", COLORS["latent"]),
            "diffusion": ("Diffusion\nProcess", COLORS["diffusion"]),
            "decoder": ("Decoder\nNetwork", COLORS["decoder"]),
            "output": ("Reconstructed\nTrajectories", COLORS["patient"]),
        }
        
        for i, (key, (label, color)) in enumerate(blocks_info.items()):
            block = VGroup()
            rect = Rectangle(
                width=2, 
                height=1.5,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
            )
            text = Text(label, color=COLORS["text"]).scale(0.4)
            text.move_to(rect.get_center())
            block.add(rect, text)
            
            # Position blocks in a logical flow
            if i == 0:  # Data
                block.to_edge(LEFT).shift(UP * 2)
            elif i == 1:  # Encoder
                block.next_to(self.blocks["data"], RIGHT, buff=1.5)
            elif i == 2:  # Latent
                block.next_to(self.blocks["encoder"], RIGHT, buff=1.5)
            elif i == 3:  # Diffusion
                block.next_to(self.blocks["latent"], DOWN, buff=1.5)
            elif i == 4:  # Decoder
                block.next_to(self.blocks["diffusion"], LEFT, buff=1.5)
            else:  # Output
                block.next_to(self.blocks["decoder"], LEFT, buff=1.5)
            
            self.blocks[key] = block
            self.add(block)
    
    def create_connections(self):
        # Create arrows between blocks
        connections = [
            ("data", "encoder"),
            ("encoder", "latent"),
            ("latent", "diffusion"),
            ("diffusion", "decoder"),
            ("decoder", "output"),
        ]
        
        for start, end in connections:
            arrow = Arrow(
                self.blocks[start].get_right(),
                self.blocks[end].get_left(),
                color=COLORS["arrow"],
                buff=0.1,
            )
            self.arrows[(start, end)] = arrow
            self.add(arrow)
    
    def highlight_block(self, block_key: str) -> Animation:
        """Create animation to highlight a specific block."""
        block = self.blocks[block_key]
        return Succession(
            block.animate.scale(1.2),
            block.animate.set_style(
                fill_opacity=0.4,
                stroke_width=3,
            )
        )
    
    def unhighlight_block(self, block_key: str) -> Animation:
        """Create animation to unhighlight a specific block."""
        block = self.blocks[block_key]
        return Succession(
            block.animate.scale(1/1.2),
            block.animate.set_style(
                fill_opacity=0.2,
                stroke_width=1,
            )
        )

class RiemannianDistribution(VGroup):
    """A class to visualize the Riemannian distribution on a manifold."""
    
    def __init__(
        self,
        manifold_func: callable,
        t_range: List[float],
        num_points: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.manifold_func = manifold_func
        self.t_range = t_range
        self.num_points = num_points
        self.create_manifold()
        self.create_metric_tensor_field()
        
    def create_manifold(self):
        self.manifold = ParametricFunction(
            self.manifold_func,
            t_range=self.t_range,
            color=COLORS["latent"],
        )
        self.add(self.manifold)
        
    def create_metric_tensor_field(self):
        """Create visualization of the metric tensor field."""
        self.metric_tensors = VGroup()
        ts = np.linspace(self.t_range[0], self.t_range[1], 20)
        
        for t in ts:
            point = self.manifold_func(t)
            # Create ellipse representing the metric tensor
            ellipse = Circle(radius=0.2)
            ellipse.move_to(point)
            # Stretch based on manifold curvature
            stretch_factor = 1 + 0.5 * np.sin(t)
            ellipse.stretch(stretch_factor, 0)
            ellipse.stretch(1/stretch_factor, 1)
            ellipse.rotate(t)
            ellipse.set_style(
                stroke_color=COLORS["latent"],
                fill_color=COLORS["latent"],
                fill_opacity=0.1,
            )
            self.metric_tensors.add(ellipse)
        
        self.add(self.metric_tensors)
    
    def sample_points(self, num_samples: int = 50) -> VGroup:
        """Sample points from the Riemannian distribution."""
        points = VGroup()
        ts = np.random.uniform(self.t_range[0], self.t_range[1], num_samples)
        
        for t in ts:
            base_point = self.manifold_func(t)
            # Add noise in the tangent space
            noise_magnitude = 0.2 * (1 + 0.5 * np.sin(t))
            noise = np.random.normal(0, noise_magnitude, 2)
            point = Dot(
                point=np.array([
                    base_point[0] + noise[0],
                    base_point[1] + noise[1],
                    0
                ]),
                color=COLORS["latent"],
                radius=0.05,
            )
            points.add(point)
        
        return points

class DiffusionProcess(VGroup):
    """A class to visualize the diffusion process."""
    
    def __init__(
        self,
        start_point: np.ndarray,
        num_steps: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.start_point = start_point
        self.num_steps = num_steps
        self.create_time_axis()
        self.create_diffusion_path()
        
    def create_time_axis(self):
        """Create the time axis for the diffusion process."""
        self.time_axis = NumberLine(
            x_range=[0, self.num_steps, 1],
            length=8,
            include_numbers=True,
            label_direction=DOWN,
        )
        self.time_label = Text("Time Steps", color=COLORS["text"]).scale(0.5)
        self.time_label.next_to(self.time_axis, DOWN)
        self.add(self.time_axis, self.time_label)
        
    def create_diffusion_path(self):
        """Create the diffusion path visualization."""
        self.points = VGroup()
        self.noise_circles = VGroup()
        self.formulas = VGroup()
        
        for t in range(self.num_steps + 1):
            # Calculate diffusion parameters
            alpha_bar_t = np.exp(-t * np.log(20) / self.num_steps)
            sigma_t = np.sqrt(1 - alpha_bar_t)
            
            # Create point
            noise = np.random.normal(0, sigma_t, 2)
            point_pos = np.array([
                self.start_point[0] * np.sqrt(alpha_bar_t) + noise[0],
                self.start_point[1] * np.sqrt(alpha_bar_t) + noise[1],
                0
            ])
            point = Dot(point=point_pos, color=COLORS["diffusion"])
            
            # Create noise circle
            circle = Circle(
                radius=sigma_t,
                color=COLORS["diffusion"],
                fill_opacity=0.1,
            )
            circle.move_to(point)
            
            # Create formula
            formula = MathTex(
                f"z_{t} = \\sqrt{{\\bar{{\\alpha}}_{t}}}z_0 + \\sqrt{{1-\\bar{{\\alpha}}_{t}}}\\epsilon",
                color=COLORS["text"]
            ).scale(0.4)
            formula.next_to(point, UP)
            
            self.points.add(point)
            self.noise_circles.add(circle)
            self.formulas.add(formula)
        
        self.add(self.points, self.noise_circles, self.formulas)

class EncoderNetwork(VGroup):
    """A class to visualize the encoder network."""
    
    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = [8, 16, 8], **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.create_network()
        self.create_distribution()
        
    def create_network(self):
        """Create the neural network visualization."""
        self.layers = VGroup()
        dims = [self.input_dim] + self.hidden_dims + [2]  # 2D latent space
        
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layer = VGroup()
            
            # Create neurons
            neurons = VGroup()
            for j in range(max(in_dim, out_dim)):
                if j < in_dim:
                    neuron = Circle(radius=0.1, color=COLORS["encoder"])
                    neuron.shift(UP * (j - in_dim/2) * 0.5)
                    neurons.add(neuron)
            
            # Position layer
            layer.add(neurons)
            if i == 0:
                layer.to_edge(LEFT)
            else:
                layer.next_to(self.layers[-1], RIGHT, buff=1)
            
            self.layers.add(layer)
        
        # Add connections between layers
        self.connections = VGroup()
        for i in range(len(self.layers) - 1):
            layer1, layer2 = self.layers[i], self.layers[i+1]
            for n1, n2 in zip(layer1[0], layer2[0]):
                connection = Line(
                    n1.get_right(),
                    n2.get_left(),
                    stroke_opacity=0.5,
                    color=COLORS["encoder"]
                )
                self.connections.add(connection)
        
        self.add(self.layers, self.connections)
    
    def create_distribution(self):
        """Create the output distribution visualization."""
        # Create mean and variance outputs
        self.mean_output = VGroup()
        self.var_output = VGroup()
        
        # Mean vector
        mean_arrow = Arrow(
            self.layers[-1].get_right(),
            self.layers[-1].get_right() + RIGHT * 2,
            color=COLORS["encoder"]
        )
        mean_label = MathTex("\\mu(x)", color=COLORS["text"]).scale(0.8)
        mean_label.next_to(mean_arrow, UP)
        self.mean_output.add(mean_arrow, mean_label)
        
        # Variance vector
        var_arrow = Arrow(
            self.layers[-1].get_right(),
            self.layers[-1].get_right() + RIGHT * 2 + DOWN,
            color=COLORS["encoder"]
        )
        var_label = MathTex("\\sigma^2(x)", color=COLORS["text"]).scale(0.8)
        var_label.next_to(var_arrow, DOWN)
        self.var_output.add(var_arrow, var_label)
        
        self.add(self.mean_output, self.var_output)

class DecoderNetwork(VGroup):
    """A class to visualize the decoder network."""
    
    def __init__(self, output_dim: int = 4, hidden_dims: List[int] = [8, 16, 8], **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.create_network()
        self.create_distribution()
        
    def create_network(self):
        """Create the neural network visualization."""
        self.layers = VGroup()
        dims = [2] + self.hidden_dims + [self.output_dim]  # 2D latent space input
        
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layer = VGroup()
            
            # Create neurons
            neurons = VGroup()
            for j in range(max(in_dim, out_dim)):
                if j < in_dim:
                    neuron = Circle(radius=0.1, color=COLORS["decoder"])
                    neuron.shift(UP * (j - in_dim/2) * 0.5)
                    neurons.add(neuron)
            
            # Position layer
            layer.add(neurons)
            if i == 0:
                layer.to_edge(LEFT)
            else:
                layer.next_to(self.layers[-1], RIGHT, buff=1)
            
            self.layers.add(layer)
        
        # Add connections between layers
        self.connections = VGroup()
        for i in range(len(self.layers) - 1):
            layer1, layer2 = self.layers[i], self.layers[i+1]
            for n1, n2 in zip(layer1[0], layer2[0]):
                connection = Line(
                    n1.get_right(),
                    n2.get_left(),
                    stroke_opacity=0.5,
                    color=COLORS["decoder"]
                )
                self.connections.add(connection)
        
        self.add(self.layers, self.connections)
    
    def create_distribution(self):
        """Create the output distribution visualization."""
        # Create reconstructed output distribution
        self.output_dist = VGroup()
        
        # Mean reconstruction
        output_arrow = Arrow(
            self.layers[-1].get_right(),
            self.layers[-1].get_right() + RIGHT * 2,
            color=COLORS["decoder"]
        )
        output_label = MathTex("\\hat{x}", color=COLORS["text"]).scale(0.8)
        output_label.next_to(output_arrow, UP)
        
        # Distribution visualization
        dist_circle = Circle(
            radius=0.5,
            color=COLORS["decoder"],
            fill_opacity=0.1
        )
        dist_circle.next_to(output_arrow, RIGHT)
        dist_label = MathTex(
            "p(x|z) = \\mathcal{N}(\\hat{x}, \\sigma^2I)",
            color=COLORS["text"]
        ).scale(0.6)
        dist_label.next_to(dist_circle, UP)
        
        self.output_dist.add(output_arrow, output_label, dist_circle, dist_label)
        self.add(self.output_dist)

def blob_manifold(t):
    """Create a blob-like manifold similar to a star/splash shape."""
    R = 2
    # Use multiple sinusoidal components to create organic shape
    r = 1.0 * (1 + 0.5 * np.sin(5*t) + 0.3 * np.cos(3*t))  # Increased base radius
    return np.array([
        r * np.cos(t),
        r * np.sin(t),
        0
    ])

def riemannian_manifold(t, s):
    """Create a filled crescent-like Riemannian manifold."""
    R = 2.0
    # s parameter controls the thickness (0 to 1)
    thickness = 2.0 * s  # Increased thickness even more
    angle = t * PI
    
    # Base curve with more pronounced shape
    x = R * np.cos(angle) * (1 + 0.3 * np.sin(2*angle))
    y = R * np.sin(angle) * (1 + 0.3 * np.sin(2*angle))
    
    # Add thickness to create filled area with smooth transition
    x += thickness * np.cos(angle + PI/2)
    y += thickness * np.sin(angle + PI/2)
    
    return np.array([x, y, 0])

def sample_riemannian_point():
    """Sample a point uniformly from the Riemannian manifold."""
    t = np.random.uniform(0, 1)  # Position along the curve
    s = np.random.uniform(0, 1)  # Position in thickness
    return riemannian_manifold(t, s)

class GUGUSReconstructionPipeline(MovingCameraScene):
    """Main scene for the GUGUS reconstruction pipeline visualization."""
    
    def __init__(self):
        super().__init__()
        self.current_mobjects = VGroup()
        self.num_points = 5  # Reduced number of points
        self.patient_data = self.generate_patient_data()
        self.i, self.j = self.select_random_point()
        self.colors = {
            "manifold": "#FF7F7F",  # Coral red for blob
            "trajectory": YELLOW,
            "selected_point": RED,
            "diffusion": BLUE,
            "reconstruction": GREEN,
            "error": WHITE,
            "title": WHITE,
            "legend": WHITE,
            "network": PURPLE,
            "latent": "#FF8C00",  # Dark orange for latent manifold
            "distribution": ORANGE
        }
        
    def generate_patient_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic patient trajectory data with strategic point placement."""
        data = {}
        
        # Define strategic positions at the vertices of the manifold
        # These are relative polar coordinates (angle, radius_factor)
        strategic_positions = [
            (0.1*PI, 0.98),    # Right vertex (even closer to edge)
            (0.5*PI, 1.8* 0.98),    # Upper right vertex
            (1.0*PI, 0.8*0.98),    # Top vertex
            (1.2*PI, 2 * 0.98),    # Upper left vertex
            (1.6*PI, 0.98),    # Left vertex
        ]
        
        points = []
        for angle, radius_factor in strategic_positions:
            # Add very small randomness to keep points at vertices
            perturbed_angle = angle + np.random.uniform(-0.05, 0.05)
            base_point = blob_manifold(perturbed_angle)
            
            # Calculate a point very close to the manifold boundary
            point = radius_factor * np.array([base_point[0], base_point[1]])
            points.append(point)
        
        data["patient_0"] = np.array(points)
        return data
    
    def select_random_point(self) -> Tuple[int, int]:
        """Select a random patient and time point."""
        patient_idx = np.random.randint(0, len(self.patient_data))
        patient_key = f"patient_{patient_idx}"
        time_idx = np.random.randint(0, len(self.patient_data[patient_key]))
        return patient_idx, time_idx

    def construct(self):
        # Scene 1: Data Manifold and Distribution
        self.show_data_manifold()
        self.wait(2)
        
        # Scene 2: Encoding and Latent Space
        self.show_encoding_process()
        self.wait(2)
        
        # Scene 3: Diffusion Process
        self.show_diffusion_process()
        self.wait(2)
        
        # Scene 4: Decoding and Reconstruction
        self.show_decoding_process()
        self.wait(2)
        
        # Scene 5: Final Comparison
        self.show_final_comparison()
        self.wait(2)
    
    def create_legend(self, items: List[Tuple[str, str]], position: np.ndarray = DOWN) -> VGroup:
        """Create a legend with given items."""
        legend = VGroup()
        for i, (text, color) in enumerate(items):
            dot = Dot(color=color, radius=0.1)
            label = Text(text, color=self.colors["legend"]).scale(0.4)
            group = VGroup(dot, label)
            group.arrange(RIGHT, buff=0.2)
            legend.add(group)
        
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.to_edge(position)
        return legend
    
    def create_trail(self, points: List[Dot], color: str, opacity: float = 0.3) -> VGroup:
        """Create a trail effect between points."""
        trail = VGroup()
        for i in range(len(points) - 1):
            line = Line(
                points[i].get_center(),
                points[i + 1].get_center(),
                color=color,
                stroke_opacity=opacity
            )
            trail.add(line)
        return trail
    
    def create_network_scheme(self, input_dim: int, hidden_dims: List[int], output_dim: int, color: str) -> VGroup:
        """Create a simple network scheme visualization."""
        network = VGroup()
        
        # Create layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i, dim in enumerate(dims):
            layer = VGroup()
            for j in range(dim):
                neuron = Circle(radius=0.1, color=color, fill_opacity=0.5)
                neuron.shift(UP * (j - (dim-1)/2) * 0.5)
                layer.add(neuron)
            layer.shift(RIGHT * i * 1.5)
            layers.append(layer)
            network.add(layer)
        
        # Create connections
        for i in range(len(layers) - 1):
            for n1 in layers[i]:
                for n2 in layers[i + 1]:
                    line = Line(
                        n1.get_right(),
                        n2.get_left(),
                        color=color,
                        stroke_opacity=0.3
                    )
                    network.add(line)
        
        return network
    
    def create_distribution_cloud(self, num_points: int, color: str, radius: float = 0.5) -> VGroup:
        """Create a cloud of points representing a distribution."""
        cloud = VGroup()
        for _ in range(num_points):
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, radius)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            point = Dot(point=np.array([x, y, 0]), color=color, radius=0.05)
            cloud.add(point)
        return cloud
    
    def show_data_manifold(self):
        """Show the blob manifold with strategic trajectories."""
        # Set initial camera position
        self.camera.frame.move_to(ORIGIN)
        self.camera.frame.set_width(14)

        # Create title
        title = Text("Data Distribution and Patient Trajectories", color=self.colors["title"])
        title.scale(0.8).to_edge(UP)
        
        # Create blob manifold (larger size)
        manifold = ParametricFunction(
            blob_manifold,
            t_range=[0, 2*PI],
            color=self.colors["manifold"],
            fill_opacity=0.3
        ).scale(2.5).to_edge(LEFT, buff=3)
        
        # Create single trajectory
        trajectories = VGroup()
        data = self.patient_data["patient_0"]
        points = VGroup()
        labels = VGroup()
        arrows = VGroup()
        
        # Create points with labels
        for time_idx, point in enumerate(data):
            # Create dot
            dot = Dot(
                point=np.array([point[0], point[1], 0]),
                color=self.colors["trajectory"] if time_idx != 2 else self.colors["selected_point"],
                radius=0.08
            )
            dot.shift(manifold.get_center())
            
            # Create label with new numbering (1 to 5)
            label = MathTex(
                f"x_i,{time_idx + 1}",
                color=self.colors["trajectory"] if time_idx != 2 else self.colors["selected_point"]
            ).scale(0.6)
            label.next_to(dot, UP + RIGHT, buff=0.1)
            
            points.add(dot)
            labels.add(label)
            
            # Create arrow to next point (except for last point)
            if time_idx < len(data) - 1:
                next_point = data[time_idx + 1]
                arrow = Arrow(
                    start=dot.get_center(),
                    end=np.array([next_point[0], next_point[1], 0]) + manifold.get_center(),
                    color=self.colors["trajectory"],
                    buff=0.2,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                arrows.add(arrow)
        
        # Create trail
        trail = self.create_trail(points, self.colors["trajectory"])
        trajectories.add(VGroup(points, labels, arrows, trail))
        
        # Create legend at bottom left
        legend_items = [
            ("Original Manifold", self.colors["manifold"]),
            ("Patient Trajectory", self.colors["trajectory"]),
            ("Selected Point", self.colors["selected_point"])
        ]
        legend = self.create_legend(legend_items, DOWN + LEFT)
        legend.move_to(DOWN * 2 + LEFT * 5.5)
        
        # Animate
        self.play(
            Create(manifold),
            Write(title)
        )
        
        # Animate trajectory with sequential point appearance
        trajectory = trajectories[0]
        points, labels, arrows, trail = trajectory
        for point, label, arrow in zip(points, labels, list(arrows) + [None]):
            self.play(
                Create(point),
                Write(label),
                run_time=0.5
            )
            if arrow:
                self.play(Create(arrow), run_time=0.3)
        
        self.play(Write(legend))
        
        # Store initial elements
        self.current_mobjects = VGroup(
            manifold, trajectories,
        )
        
        self.manifold = manifold
        self.remove(legend)
        # Camera movement for transition
        self.play(
            self.camera.frame.animate.scale(2).shift(RIGHT * 8),
            FadeOut(title),
            run_time=2
        )
    
        self.remove(title)

    def create_metric_ellipse(self, center, scale=1.0, angle=None):
        """Create an ellipse representing the local Riemannian metric."""
        # Create base ellipse
        ellipse = Ellipse(
            width=0.4 * scale,
            height=0.2 * scale,
            color=ORANGE,
            fill_opacity=0.1,
            stroke_opacity=0.3
        )
        ellipse.move_to(center)
        
        # If no angle provided, use position-dependent angle
        if angle is None:
            # Calculate angle based on position relative to origin
            pos = center - ORIGIN
            angle = np.arctan2(pos[1], pos[0])
            # Add some variation
            angle += np.random.uniform(-PI/6, PI/6)
        
        # Rotate ellipse
        ellipse.rotate(angle)
        
        return ellipse

    def show_encoding_process(self):
        """Show the encoding process and latent space."""
        # Keep previous elements but fade them
        self.play(*[mob.animate.set_opacity(0.3) for mob in self.current_mobjects])
        
        # Create title
        title = Text("Latent Space Representation", color=self.colors["title"])
        title.scale(1.3).to_edge(UP).move_to(UP * 6 + RIGHT * 4)

        # Create encoder network
        encoder = self.create_network_scheme(2, [3], 2, self.colors["network"])
        encoder.scale(1.2).next_to(self.current_mobjects[1], RIGHT, buff=3)
        
        # Add network labels
        input_label = MathTex("x", color=self.colors["network"]).next_to(encoder[0], LEFT)
        output_label = MathTex("z", color=self.colors["network"]).next_to(encoder[-1], RIGHT)
        
        # Add encoder distribution law
        encoder_law = MathTex(
            "q_\\phi(z|x)", 
            color=self.colors["network"]
        ).scale(0.8)
        encoder_law.next_to(encoder, DOWN, buff=0.5)
        
        # Create filled Riemannian manifold
        num_samples = 150
        latent_points = VGroup()
        metric_tensors = VGroup()  # Group for metric ellipses
        
        # Sample points from the manifold with guaranteed inside placement
        for _ in range(num_samples):
            point = self.sample_point_in_latent_manifold()
            dot = Dot(point=point, color=self.colors["latent"], radius=0.02)
            latent_points.add(dot)
            
            # Create metric tensor ellipse for each point
            ellipse = self.create_metric_ellipse(dot.get_center(), scale=1.0)
            metric_tensors.add(ellipse)
        
        # Create manifold outline
        t_values = np.linspace(0, 1, 200)
        outline_points = []
        
        for t in t_values:
            outline_points.append(riemannian_manifold(t, 1))
        for t in reversed(t_values):
            outline_points.append(riemannian_manifold(t, 0))
        
        outline = Polygon(
            *outline_points,
            color=self.colors["latent"],
            fill_opacity=0.2,
            stroke_width=2
        ).scale(2.0)
        
        # Scale up the points to match the outline
        latent_points.scale(2.0)
        metric_tensors.scale(2.0)
        
        # Position latent space
        latent_group = VGroup(outline, latent_points, metric_tensors)
        latent_group.next_to(encoder, RIGHT, buff=2)
        
        # Create legend
        legend_items = [
            ("Latent Manifold", self.colors["latent"]),
            ("Sampled Points", self.colors["distribution"]),
            ("Local Metric G(z)", ORANGE)
        ]
        legend = self.create_legend(legend_items)
        legend.next_to(latent_group, DOWN, buff=1)
        
        # Animate
        self.play(Write(title))
        self.play(Create(encoder), run_time=1.5)
        self.play(Write(encoder_law))
        self.play(
            Write(input_label),
            Write(output_label),
            Create(outline),
            run_time=1.5
        )
        
        # Animate points and metrics with slight delay
        self.play(LaggedStart(*[Create(p) for p in latent_points], lag_ratio=0.01))
        self.play(LaggedStart(*[Create(m) for m in metric_tensors], lag_ratio=0.01))
        
        self.play(Write(legend))
        
        # Store elements
        new_mobjects = VGroup(
            encoder, input_label, output_label,
            encoder_law, outline, latent_points,
            metric_tensors, legend
        )
        self.current_mobjects = VGroup(*self.current_mobjects, *new_mobjects)
        self.latent_group = latent_group
        
        self.remove(title)
    
    def sample_point_in_latent_manifold(self):
        """Sample a point that's guaranteed to be inside the latent manifold."""
        while True:
            t = np.random.uniform(0, 1)
            s = np.random.beta(2, 2)  # This creates more points in the middle
            point = riemannian_manifold(t, s)
            
            # Check if point is inside the manifold by comparing with boundary
            boundary_up = riemannian_manifold(t, 1)
            boundary_down = riemannian_manifold(t, 0)
            
            # Calculate if point is between boundaries
            up_vec = boundary_up - boundary_down
            point_vec = point - boundary_down
            
            # Project point onto up vector
            proj = np.dot(point_vec[:2], up_vec[:2]) / np.dot(up_vec[:2], up_vec[:2])
            
            # If projection is between 0 and 1, point is inside
            if 0 <= proj <= 1:
                return point
    
    def sample_trajectory_points(self, num_points, start_t=None):
        """Sample a sequence of points that form a continuous trajectory in the manifold."""
        points = []
        
        # If no start_t provided, choose random
        if start_t is None:
            start_t = np.random.uniform(0.2, 0.8)  # Avoid edges
        
        current_t = start_t
        
        for i in range(num_points):
            # Sample s with stronger preference for middle values
            s = np.random.beta(3, 3)  # More concentrated in the middle
            point = riemannian_manifold(current_t, s)
            points.append(point)
            
            # Move t smoothly along the manifold with some randomness
            # but ensuring we stay within bounds and maintain continuity
            delta_t = np.random.uniform(0.1, 0.15)  # Smaller steps
            current_t = (current_t + delta_t) % 1.0
            
            # Keep points away from edges
            if current_t < 0.1:
                current_t = 0.1
            elif current_t > 0.9:
                current_t = 0.9
        
        return points

    def show_diffusion_process(self):
        """Show the diffusion process in the latent space."""
        # Keep previous elements but fade them more
        self.play(*[mob.animate.set_opacity(0.1) for mob in self.current_mobjects])
        
        # Define colors for past and future
        past_color = "#4169E1"  # Royal Blue
        future_color = "#32CD32"  # Lime Green
        
        # Create diffusion process title
        title = Text("Diffusion Process in Latent Space", color=self.colors["title"])
        title.scale(1.3).move_to(UP * 6 + RIGHT * 4)
        
        # Create diffusion equations
        equations = VGroup()
        
        # Past process equation (blue)
        past_eq = MathTex(
            "p_\\theta(z_{t-1}|z_t) = \\mathcal{N}(z_{t-1}; \\mu_\\theta(z_t, t), \\Sigma_\\theta(z_t, t))",
            color=past_color
        ).scale(0.8)
        
        # Future process equation (green)
        future_eq = MathTex(
            "q(z_t|z_{t-1}) = \\mathcal{N}(z_t; \\sqrt{1-\\beta_t}z_{t-1}, \\beta_t I)",
            color=future_color
        ).scale(0.8)
        
        equations.add(past_eq, future_eq)
        equations.arrange(DOWN, buff=0.5)
        equations.move_to(DOWN * 3 + RIGHT * 6)
        
        # Find the latent manifold outline for reference
        outline = [mob for mob in self.current_mobjects if isinstance(mob, Polygon)][0]
        
        # Sample a continuous trajectory for diffusion
        num_points = 5  # Total points in trajectory
        trajectory_points = self.sample_trajectory_points(num_points)
        trajectory_points = [np.array([p[0], p[1], 0]) * 2.0 for p in trajectory_points]  # Scale points
        
        # Create dots and labels for trajectory
        dots = VGroup()
        labels = VGroup()
        future_arrows = VGroup()
        past_arrows = VGroup()
        
        # After creating each point in the trajectory, add its metric tensor
        trajectory_metrics = VGroup()
        
        # Create diffusion points with metrics
        for i, point in enumerate(trajectory_points):
            # Create the point (existing code)
            dot = Dot(
                point=point + outline.get_center(),
                color=YELLOW if i != 2 else RED,
                radius=0.08 if i == 2 else 0.06
            )
            # Update label format from z_{i,n} to z_in
            label = MathTex(f"z_{{{self.i}{i+1}}}", color=YELLOW if i != 2 else RED).scale(0.6)
            label.next_to(dot, UP)
            
            dots.add(dot)
            labels.add(label)
            
            # Add metric tensor ellipse
            # Scale and stretch based on diffusion time
            if i < 2:  # Past points
                scale = 0.8 + 0.2 * i  # Gradually increase size
                stretch = 1.0 - 0.2 * i  # More compressed for earlier points
            elif i > 2:  # Future points
                scale = 1.0 + 0.2 * (i-2)  # Gradually increase size
                stretch = 1.0 + 0.2 * (i-2)  # More stretched for later points
            else:  # Current point
                scale = 1.0
                stretch = 1.0
            
            ellipse = self.create_metric_ellipse(dot.get_center(), scale=scale)
            ellipse.stretch(stretch, 0)  # Stretch in x direction
            
            # Color based on past/present/future
            if i < 2:
                ellipse.set_color(past_color)
            elif i > 2:
                ellipse.set_color(future_color)
            else:
                ellipse.set_color(self.colors["selected_point"])
            
            trajectory_metrics.add(ellipse)
            
            # Add label for the current point's metric
            if i == 2:
                metric_label = MathTex("G(z_{i,3})", color=WHITE).scale(0.6)
                metric_label.next_to(ellipse, UP + RIGHT, buff=0.1)
                trajectory_metrics.add(metric_label)
            
            # Add future arrows (from current to next)
            if i >= 2 and i < len(trajectory_points) - 1:
                next_point = trajectory_points[i + 1]
                arrow = Arrow(
                    start=dot.get_center(),
                    end=np.array([next_point[0], next_point[1], 0]) + outline.get_center(),
                    color=future_color,
                    buff=0.2,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                future_arrows.add(arrow)
        
        # Add past arrows (from current to previous)
        for i in range(2, 0, -1):
            current_point = dots[i].get_center()
            prev_point = dots[i-1].get_center()
            arrow = Arrow(
                start=current_point,
                end=prev_point,
                color=past_color,
                buff=0.2,
                max_tip_length_to_length_ratio=0.15,
                stroke_width=2
            )
            past_arrows.add(arrow)
        
        # Create legend
        legend_items = [
            ("Current Point", self.colors["selected_point"]),
            ("Future Process", future_color),
            ("Past Process", past_color),
            ("Local Metric G(z)", WHITE)
        ]
        legend = self.create_legend(legend_items)
        legend.next_to(equations, DOWN, buff=1)
        
        # Animate
        self.play(Write(title))
        self.play(Write(equations))
        
        # Animate trajectory appearance
        self.play(Create(dots[2]), Write(labels[2]))  # Start with middle point
        
        # Animate future points with green equation highlight
        self.play(future_eq.animate.set_color(future_color).scale(1.1))
        for i in range(3, len(dots)):
            self.play(
                Create(dots[i]),
                Write(labels[i]),
                Create(future_arrows[i-3]),  # Adjusted index for future arrows
                run_time=0.8
            )
        self.play(future_eq.animate.scale(1/1.1))
        
        # Animate past points with blue equation highlight
        self.play(past_eq.animate.set_color(past_color).scale(1.1))
        for i, idx in enumerate(range(1, -1, -1)):
            self.play(
                Create(dots[idx]),
                Write(labels[idx]),
                Create(past_arrows[i]),  # Use i for past arrows index
                run_time=0.8
            )
        self.play(past_eq.animate.scale(1/1.1))
        
        self.play(Write(legend))
        
        # Store new elements
        new_mobjects = VGroup(
            equations, dots, labels, future_arrows, past_arrows, legend, trajectory_metrics
        )
        self.current_mobjects = VGroup(*self.current_mobjects, *new_mobjects)
        #self.play(*[mob.animate.set_opacity(1) for mob in self.current_mobjects])
        self.remove(title)
    
    def create_reconstruction(self, center_position=None, scale=2.5):
        """Create a reconstructed manifold with correlated noise for both points and manifold."""
        # Create noisy manifold
        def noisy_blob_manifold(t):
            base_point = blob_manifold(t)
            noise = np.array([
                np.random.normal(0, 0.05),
                np.random.normal(0, 0.05),
                0
            ])
            return base_point + noise
        
        reconstructed_manifold = ParametricFunction(
            noisy_blob_manifold,
            t_range=[0, 2*PI],
            color=self.colors["reconstruction"],
            fill_opacity=0.2
        ).scale(scale)
        
        # Position the manifold first before creating points
        if center_position is not None:
            reconstructed_manifold.move_to(center_position)
        
        # Create points with correlated noise
        reconstructed_points = VGroup()
        reconstructed_labels = VGroup()
        reconstructed_arrows = VGroup()
        
        # Get original points positions
        data = self.patient_data["patient_0"]
        noisy_points = []
        
        # Calculate manifold radius for point containment
        base_radius = 2.0  # Base radius of blob_manifold
        max_noise = 0.05  # Same as manifold noise
        manifold_radius = (base_radius + max_noise) * scale  # Scaled radius with noise
        
        # First generate all noisy points to ensure trajectory continuity
        for point in data:
            point_3d = np.array([point[0], point[1], 0])
            noise_scale = 0.15  # Noise scale for points
            max_attempts = 10  # Maximum attempts to generate valid point
            
            valid_noise = None
            for _ in range(max_attempts):
                if noisy_points:  # If we have previous points, correlate noise with last point
                    last_noise = noisy_points[-1] - np.array([point[0], point[1], 0])
                    new_noise = np.array([
                        np.random.normal(0.3 * last_noise[0], noise_scale),
                        np.random.normal(0.3 * last_noise[1], noise_scale),
                        0
                    ])
                else:
                    new_noise = np.array([
                        np.random.normal(0, noise_scale),
                        np.random.normal(0, noise_scale),
                        0
                    ])
                
                # Check if point would be inside manifold relative to manifold's center
                test_point = point_3d + new_noise
                distance_from_center = np.linalg.norm(test_point[:2])
                if distance_from_center <= manifold_radius:
                    valid_noise = new_noise
                    break
            
            # If no valid noise found, scale down the last attempted noise
            if valid_noise is None:
                scale_factor = manifold_radius / distance_from_center
                valid_noise = new_noise * scale_factor * 0.95  # 5% margin
            
            noisy_point = point_3d + valid_noise
            noisy_points.append(noisy_point)
        
        # Create points and arrows relative to manifold's current position
        manifold_center = reconstructed_manifold.get_center()
        
        # Create points and arrows
        for i, (original_point, noisy_point) in enumerate(zip(data, noisy_points)):
            # Create dot at the correct position relative to manifold center
            dot = Dot(
                point=manifold_center + noisy_point,
                color=YELLOW if i != 2 else RED,
                radius=0.08 if i == 2 else 0.06
            )
            
            label = MathTex(f"\\hat{{x}}_{{i,{i+1}}}", color=YELLOW if i != 2 else RED).scale(0.6)
            label.next_to(dot, UP + RIGHT, buff=0.1)
            
            reconstructed_points.add(dot)
            reconstructed_labels.add(label)
            
            if i < len(data) - 1:
                next_point = noisy_points[i + 1]
                arrow = Arrow(
                    start=dot.get_center(),
                    end=manifold_center + next_point,
                    color=YELLOW,
                    buff=0.2,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                reconstructed_arrows.add(arrow)
        
        # Group everything together so it moves as one unit
        reconstruction_group = VGroup(
            reconstructed_manifold,
            reconstructed_points,
            reconstructed_labels,
            reconstructed_arrows
        )
        
        return reconstruction_group

    def show_decoding_process(self):
        """Show the decoding process and reconstruction."""
        # Find the latent manifold and points (more robust search)
        outline = None
        latent_points = None
        encoder = None
        encoder_law = None
        elements_to_restore = []
        
        # Search through current mobjects and store elements to restore
        for mob in self.current_mobjects:
            if isinstance(mob, VGroup):
                # Check if this is the latent_group
                if mob == self.latent_group:
                    elements_to_restore.append(mob)
                    # Also add its components
                    for submob in mob:
                        elements_to_restore.append(submob)
                        if isinstance(submob, VGroup):
                            for subsubmob in submob:
                                elements_to_restore.append(subsubmob)
                # Check other VGroup elements
                for submob in mob:
                    if isinstance(submob, VGroup) and any(isinstance(s, Circle) for s in submob):
                        if submob.get_color() == self.colors["network"]:
                            encoder = submob
                            elements_to_restore.append(submob)
                    elif isinstance(submob, MathTex) and "phi" in submob.tex_string:
                        encoder_law = submob
                        elements_to_restore.append(submob)
        
        # Restore original opacities for found elements
        restore_animations = []
        for elem in elements_to_restore:
            original_opacity = self.original_opacities.get(hash(elem), 1.0)
            if isinstance(elem, VMobject):  # Check if element supports opacity
                restore_animations.append(elem.animate.set_opacity(original_opacity))
        
        # Fade other elements
        fade_animations = [
            mob.animate.set_opacity(0.3) 
            for mob in self.current_mobjects 
            if mob not in elements_to_restore and isinstance(mob, VMobject)
        ]
        
        if restore_animations or fade_animations:
            self.play(
                *restore_animations,
                *fade_animations,
                run_time=1
            )
        
        # Then dezoom
        self.play(
            self.camera.frame.animate.scale(1.0).shift(RIGHT * 16),
            run_time=2
        )
        
        # Create title in the right position
        title = Text("Decoding Process", color=self.colors["title"])
        title.scale(1.8).move_to(UP * 9 + RIGHT * 16)
        
        # Create decoder network - position relative to latent space
        decoder = self.create_network_scheme(2, [3], 2, self.colors["network"])
        decoder.scale(1.2)
        decoder.next_to(self.latent_group, RIGHT, buff=3)
        
        # Add network labels
        input_label = MathTex("z", color=self.colors["network"]).next_to(decoder[0], LEFT, buff=0.5)
        output_label = MathTex("\\hat{x}", color=self.colors["network"]).next_to(decoder[-1], RIGHT, buff=0.5)
        
        # Add decoder distribution law
        decoder_law = MathTex(
            "p_\\theta(x|z) = \\mathcal{N}(x; \\mu_\\theta(z), \\sigma^2_\\theta(z)I)", 
            color=self.colors["network"]
        ).scale(0.8)
        decoder_law.next_to(decoder, DOWN, buff=0.5)
        
        # Create reconstruction and position it
        reconstruction_group = self.create_reconstruction(scale=2.5)
        reconstruction_group.next_to(decoder, RIGHT, buff=3)
        
        reconstructed_manifold = reconstruction_group[0]
        reconstructed_points = reconstruction_group[1]
        reconstructed_labels = reconstruction_group[2]
        reconstructed_arrows = reconstruction_group[3]
        
        # Create legend with updated colors
        legend_items = [
            ("Decoder Network", self.colors["network"]),
            ("Reconstructed Manifold", self.colors["reconstruction"]),
            ("Reconstructed Trajectory", YELLOW)
        ]
        legend = self.create_legend(legend_items)
        legend.next_to(reconstructed_manifold, DOWN, buff=1)
        
        # Animate
        self.play(Write(title))
        
        # Animate decoder network and law
        self.play(
            Create(decoder),
            Write(input_label),
            Write(output_label)
        )
        self.play(Write(decoder_law))
        
        # Animate reconstructed manifold
        self.play(Create(reconstructed_manifold))
        
        # Animate points and arrows sequentially with smooth transitions
        for i in range(len(reconstructed_points)):
            self.play(
                Create(reconstructed_points[i]),
                Write(reconstructed_labels[i]),
                run_time=0.5
            )
            if i < len(reconstructed_arrows):
                self.play(Create(reconstructed_arrows[i]), run_time=0.3)
        
        self.play(Write(legend))
        
        # Store new elements
        new_mobjects = VGroup(
            title, decoder, input_label, output_label,
            decoder_law, reconstructed_manifold,
            reconstructed_points, reconstructed_labels,
            reconstructed_arrows, legend
        )
        self.current_mobjects = VGroup(*self.current_mobjects, *new_mobjects)
    
    
    def show_final_comparison(self):
        """Show the final comparison between original and reconstructed trajectories."""
        # First, fade out all previous elements except the manifolds and trajectories
        fade_animations = [
            mob.animate.set_opacity(0) 
            for mob in self.current_mobjects 
            if isinstance(mob, VMobject)
        ]
        self.play(*fade_animations)
        
        # Dezoom and recenter camera
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(ORIGIN),
            run_time=2
        )
        
        # Create original manifold on the left
        original_manifold = ParametricFunction(
            blob_manifold,
            t_range=[0, 2*PI],
            color=self.colors["manifold"],
            fill_opacity=0.3
        ).scale(2.0).to_edge(LEFT, buff=0.3)
        
        # Create reconstruction and position it
        reconstruction_group = self.create_reconstruction(scale=2.0)
        reconstruction_group.to_edge(RIGHT, buff=0.3)
        
        reconstructed_manifold = reconstruction_group[0]
        reconstructed_points = reconstruction_group[1]
        reconstructed_labels = reconstruction_group[2]
        reconstructed_arrows = reconstruction_group[3]
        
        # Create titles for each manifold
        original_title = Text("Original Distribution", color=WHITE).scale(0.8)
        original_title.next_to(original_manifold, UP, buff=0.4)
        
        reconstructed_title = Text("Reconstructed Distribution", color=WHITE).scale(0.8)
        reconstructed_title.next_to(reconstructed_manifold, UP, buff=0.5)
        

        # Create original trajectory points
        original_points = VGroup()
        original_labels = VGroup()
        original_arrows = VGroup()
        
        # Create original points
        data = self.patient_data["patient_0"]
        for i, point in enumerate(data):
            dot = Dot(
                point=np.array([point[0], point[1], 0]),
                color=YELLOW if i != 2 else RED,
                radius=0.08 if i == 2 else 0.06
            )
            dot.move_to(original_manifold.get_center() + np.array([point[0], point[1], 0]))
            label = MathTex(f"x_{{i,{i+1}}}", color=YELLOW if i != 2 else RED).scale(0.6)
            label.next_to(dot, UP + RIGHT, buff=0.1)
            
            original_points.add(dot)
            original_labels.add(label)
            
            if i < len(data) - 1:
                next_point = data[i + 1]
                arrow = Arrow(
                    start=dot.get_center(),
                    end=original_manifold.get_center() + np.array([next_point[0], next_point[1], 0]),
                    color=YELLOW,
                    buff=0.2,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                original_arrows.add(arrow)
        
        # Create legend
        legend_items = [
            ("Trajectory Points", YELLOW),
            ("Selected Point", RED),
            ("Original Manifold", self.colors["manifold"]),
            ("Reconstructed Manifold", self.colors["reconstruction"])
        ]
        legend = self.create_legend(legend_items)
        legend.to_edge(DOWN)
        
        # Animate everything
        self.play(
            Create(original_manifold),
            Create(reconstructed_manifold),
            Write(original_title),
            Write(reconstructed_title)
        )
        
        # Animate original trajectory
        for i in range(len(original_points)):
            self.play(
                Create(original_points[i]),
                Write(original_labels[i]),
                run_time=0.5
            )
            if i < len(original_arrows):
                self.play(Create(original_arrows[i]), run_time=0.3)
        
        # Animate reconstructed trajectory
        for i in range(len(reconstructed_points)):
            self.play(
                Create(reconstructed_points[i]),
                Write(reconstructed_labels[i]),
                run_time=0.5
            )
            if i < len(reconstructed_arrows):
                self.play(Create(reconstructed_arrows[i]), run_time=0.3)
        
        # Add error visualization for the selected point (point 3)
        error_arrow = Arrow(
            start=original_points[2].get_center(),
            end=reconstructed_points[2].get_center(),
            color=WHITE,
            buff=0.1,
            stroke_width=2
        )
        error_label = MathTex(
            "\\|x_{i,3} - \\hat{x}_{i,3}\\|",
            color=WHITE
        ).scale(0.6)
        error_label.next_to(error_arrow.get_center(), UP)
        
        self.play(
            Create(error_arrow),
            Write(error_label)
        )
        
        self.play(Write(legend))
        
        # Store final state
        self.current_mobjects = VGroup(
            original_manifold, reconstructed_manifold,
            original_title, reconstructed_title,
            original_points, original_labels, original_arrows,
            reconstructed_points, reconstructed_labels, reconstructed_arrows,
            error_arrow, error_label, legend
        ) 