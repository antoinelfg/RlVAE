from manim import *
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from colour import Color
from ManimGUGUSReconstruction import GUGUSReconstructionPipeline, blob_manifold

# Configuration settings
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.media_width = "100%"
config.background_color = BLACK
config.frame_height = 8.0
config.frame_width = 14.222222222222221
config.output_file = "GUGUSTrainingVisualization"
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
    "loss": "#F44336",     # Red for loss curves
    "kl": "#4CAF50",       # Green for KL divergence
    "recon": "#2196F3",    # Blue for reconstruction loss
    "metric_reg": "#FFC107" # Amber for metric regularization
}

# PLACEMENT: Top center of the scene
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

# PLACEMENT: Left side of the scene, showing the original data manifold
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

# PLACEMENT: Center of the scene, showing the diffusion process
class DiffusionProcess(VMobject):
    """A class to visualize the bidirectional diffusion process on URIEM."""
    
    def __init__(
        self,
        start_point: np.ndarray,
        num_steps: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.start_point = start_point
        self.num_steps = num_steps
        self.points_group = VGroup()
        self.create_diffusion_path()
        self.add_uriem_label()
    
    def create_diffusion_path(self):
        """Create the bidirectional diffusion path visualization."""
        self.points_group = VGroup()
        self.noise_circles = VGroup()
        self.formulas = VGroup()
        self.forward_arrows = VGroup()
        
        # Define colors for past and future points
        past_color = "#FF6B6B"  # Light red for past
        future_color = "#4ECDC4"  # Turquoise for future
        present_color = COLORS["highlight"]  # Yellow for present
        
        # Initialize lists for forward points
        points_forward = []
        
        # Create center point in yellow first
        center_point = Dot(point=self.start_point, color=present_color)
        self.points_group.add(center_point)
        
        for t in range(self.num_steps + 1):
            # Forward diffusion parameters with increased spread
            alpha_bar_t = np.exp(-t * np.log(20) / self.num_steps)
            sigma_t = np.sqrt(1 - alpha_bar_t) * 2.0  # Increased spread factor
            
            # Create forward point with more spread
            noise = np.random.normal(0, sigma_t, 2)
            point_pos_forward = np.array([
                self.start_point[0] * np.sqrt(alpha_bar_t) + noise[0],
                self.start_point[1] * np.sqrt(alpha_bar_t) + noise[1],
                0
            ])
            point_forward = Dot(point=point_pos_forward, color=future_color)
            points_forward.append(point_forward)
            
            # Create smaller noise circle
            circle = Circle(
                radius=sigma_t * 0.3,  # Reduced size
                color=future_color if t > 0 else past_color,
                fill_opacity=0.1,
            )
            circle.move_to(point_forward)
            self.noise_circles.add(circle)
            
            # Create formula
            formula = MathTex(
                f"z_{t} = \\sqrt{{\\bar{{\\alpha}}_{t}}}z_0 + \\sqrt{{1-\\bar{{\\alpha}}_{t}}}\\epsilon",
                color=COLORS["text"]
            ).scale(0.4)
            formula.next_to(point_forward, UP)
            self.formulas.add(formula)
        
        # Add points to main group
        for point in points_forward:
            self.points_group.add(point)
        
        # Create forward arrows with appropriate colors
        for i in range(len(points_forward) - 1):
            # Forward arrow (future)
            forward_arrow = Arrow(
                points_forward[i].get_center(),
                points_forward[i + 1].get_center(),
                color=future_color,
                buff=0.1,
            )
            self.forward_arrows.add(forward_arrow)
        
        # Add all components
        self.add(
            self.points_group,
            self.noise_circles,
            self.formulas,
            self.forward_arrows
        )
    
    def add_uriem_label(self):
        """Add URIEM diffusion label."""
        label = Text("Diffusion on URIEM", color=COLORS["diffusion"]).scale(0.5)
        label.next_to(self.points_group, DOWN, buff=10)
        self.add(label)

# PLACEMENT: Right side of the scene, showing the encoder network
class EncoderNetwork(VGroup):
    """A class to visualize the encoder network during training."""
    
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
        mean_label = MathTex("\\mu_\\phi(x)", color=COLORS["text"]).scale(0.8)
        mean_label.next_to(mean_arrow, UP)
        self.mean_output.add(mean_arrow, mean_label)
        
        # Variance vector
        var_arrow = Arrow(
            self.layers[-1].get_right(),
            self.layers[-1].get_right() + RIGHT * 2 + DOWN,
            color=COLORS["encoder"]
        )
        var_label = MathTex("\\sigma^2_\\phi(x)", color=COLORS["text"]).scale(0.8)
        var_label.next_to(var_arrow, DOWN)
        self.var_output.add(var_arrow, var_label)
        
        # Add q_phi label
        q_phi_label = MathTex("q_\\phi(z|x)", color=COLORS["encoder"]).scale(0.8)
        q_phi_label.next_to(self.layers[-1], DOWN, buff=0.5)
        self.add(q_phi_label)
        
        self.add(self.mean_output, self.var_output)

# PLACEMENT: Center of the scene, showing the evolving metric field
class RiemannianMetricField(VGroup):
    """A class to visualize the evolving Riemannian metric tensor field."""
    
    def __init__(
        self,
        manifold_points: np.ndarray,
        num_tensors: int = 20,
        scale: float = 1.0,
        epoch: int = 0,
        max_epochs: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.manifold_points = manifold_points
        self.num_tensors = num_tensors
        self.scale = scale
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.create_metric_tensors()
    
    def create_metric_tensor(self, point: np.ndarray, progress: float) -> Ellipse:
        """Create a single metric tensor ellipse."""
        # Early in training: more chaotic and larger ellipses
        if self.epoch == 0:
            width = 0.6 * self.scale * (1 + np.random.uniform(-0.3, 0.3))
            height = 0.4 * self.scale * (1 + np.random.uniform(-0.3, 0.3))
            random_angle = np.random.uniform(0, 2*PI)
        else:
            # As training progresses, ellipses become more structured
            progress = self.epoch / self.max_epochs
            base_width = 0.4 * self.scale
            base_height = 0.2 * self.scale
            width = base_width * (1 + 0.3 * (1 - progress))
            height = base_height * (1 + 0.3 * (1 - progress))
            
            # Calculate position-based angle with decreasing randomness
            pos = point - ORIGIN
            base_angle = np.arctan2(pos[1], pos[0])
            random_component = np.random.uniform(-PI/3, PI/3) * (1 - progress)
            random_angle = base_angle + random_component
        
        ellipse = Ellipse(
            width=width,
            height=height,
            color=COLORS["latent"],
            fill_opacity=0.1,
            stroke_opacity=0.3
        )
        ellipse.move_to(point)
        ellipse.rotate(random_angle)
        
        return ellipse
    
    def create_metric_tensors(self):
        """Create visualization of the metric tensor field."""
        indices = np.random.choice(len(self.manifold_points), self.num_tensors, replace=False)
        progress = self.epoch / self.max_epochs if self.max_epochs > 0 else 0
        
        for idx in indices:
            point = self.manifold_points[idx]
            ellipse = self.create_metric_tensor(point, progress)
            self.add(ellipse)

# PLACEMENT: Bottom of the scene, showing the loss components
class LossVisualization(VGroup):
    """A class to visualize the GUGUS loss components with URIEM emphasis."""
    
    def __init__(self, max_width: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.max_width = max_width
        self.create_loss_components()
        self.add_no_z1_label()
    
    def add_no_z1_label(self):
        """Add label emphasizing no z1 ~ N(0,1) sampling."""
        label = Text("No z1 ~ N(0,1)", color=COLORS["text"]).scale(0.5)
        cross = Line(
            label.get_corner(UL),
            label.get_corner(DR),
            color=COLORS["loss"],
            stroke_width=2
        )
        cross2 = Line(
            label.get_corner(UR),
            label.get_corner(DL),
            color=COLORS["loss"],
            stroke_width=2
        )
        label_group = VGroup(label, cross, cross2)
        label_group.next_to(self.elbo_formula, UP, buff=0.5)
        self.add(label_group)
    
    def create_loss_components(self):
        """Create visualization of loss components with URIEM emphasis."""
        # Create ELBO formula with URIEM-specific terms
        self.elbo_formula = MathTex(
            "\\mathcal{L}_{\\text{GUGUS}} &= \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] \\\\",
            "&- D_{KL}(q_\\phi(z|x) \\| p(z)) \\\\",
            "&- \\lambda \\log |G(z)| \\\\",
            "&\\text{(on URIEM)}"
        ).scale(0.8)
        
        # Create loss bars with URIEM context
        self.recon_bar = self.create_loss_bar(COLORS["recon"], "Reconstruction")
        self.kl_bar = self.create_loss_bar(COLORS["kl"], "KL Divergence")
        self.metric_bar = self.create_loss_bar(COLORS["metric_reg"], "Metric Reg.")
        
        # Arrange components
        bars = VGroup(self.recon_bar, self.kl_bar, self.metric_bar)
        bars.arrange(DOWN, buff=0.3)
        bars.next_to(self.elbo_formula, DOWN, buff=0.5)
        
        self.add(self.elbo_formula, bars)
    
    def create_loss_bar(self, color: str, label: str) -> VGroup:
        """Create a single loss bar with label and URIEM context."""
        bar = VGroup()
        
        # Background bar
        bg = Rectangle(
            width=self.max_width,
            height=0.3,
            fill_color=COLORS["background"],
            fill_opacity=0.3,
            stroke_color=color
        )
        
        # Value bar (initially empty)
        value = Rectangle(
            width=0,
            height=0.3,
            fill_color=color,
            fill_opacity=0.5,
            stroke_width=0
        ).align_to(bg, LEFT)
        
        # Label with URIEM context
        text = Text(label, color=color).scale(0.4)
        if label == "Metric Reg.":
            text = Text("Metric Reg. (URIEM)", color=color).scale(0.4)
        text.next_to(bg, LEFT)
        
        bar.add(bg, value, text)
        return bar
    
    def update_values(self, progress: float, noise: float = 0.1):
        """Update loss bar values based on training progress with URIEM emphasis."""
        # Reconstruction loss: starts high, decreases
        recon_value = (1.0 - 0.7 * progress) + noise * np.random.random()
        self.update_bar(self.recon_bar[1], recon_value)
        
        # KL divergence: starts low, increases slightly then stabilizes
        kl_value = (0.3 + 0.2 * progress - 0.1 * progress**2) + noise * np.random.random()
        self.update_bar(self.kl_bar[1], kl_value)
        
        # Metric regularization: gradually decreases with URIEM emphasis
        metric_value = (0.5 - 0.3 * progress) + noise * np.random.random()
        self.update_bar(self.metric_bar[1], metric_value)
        
        # Update formula highlighting
        if progress > 0.5:
            self.elbo_formula[3].set_color(COLORS["highlight"])
    
    def update_bar(self, bar: Rectangle, value: float):
        """Update a single loss bar's width."""
        bar.become(
            Rectangle(
                width=self.max_width * value,
                height=0.3,
                fill_color=bar.get_color(),
                fill_opacity=0.5,
                stroke_width=0
            ).align_to(bar, LEFT)
        )

# PLACEMENT: Right side of the scene, showing the decoder network
class DecoderNetwork(VGroup):
    """A class to visualize the decoder network during training."""
    
    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [4, 8, 4], **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.create_network()
        self.create_distribution()
    
    def create_network(self):
        """Create the neural network visualization."""
        self.layers = VGroup()
        dims = [self.input_dim] + self.hidden_dims + [2]  # 2D output space
        
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
        # Create mean and variance outputs
        self.output_dist = VGroup()
        
        # Mean reconstruction
        output_arrow = Arrow(
            self.layers[-1].get_right(),
            self.layers[-1].get_right() + RIGHT * 2,
            color=COLORS["decoder"]
        )
        output_label = MathTex("\\hat{x}", color=COLORS["text"]).scale(0.8)
        output_label.next_to(output_arrow, UP)
        
        # Add p_theta label
        p_theta_label = MathTex("p_\\theta(x|z)", color=COLORS["decoder"]).scale(0.8)
        p_theta_label.next_to(self.layers[-1], DOWN, buff=0.5)
        
        self.output_dist.add(output_arrow, output_label, p_theta_label)
        self.add(self.output_dist)

# PLACEMENT: Center of the scene, showing the latent space
class LatentSpace(VMobject):
    """A class to visualize the evolving latent space during training with URIEM emphasis."""
    
    def __init__(
        self,
        width: float = 8.0,  # Increased width to accommodate diffusion
        height: float = 8.0,  # Increased height to accommodate diffusion
        num_points: int = 20,  # Reduced number of points
        epoch: int = 0,
        max_epochs: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._width = width
        self._height = height
        self.num_points = num_points
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.points_group = VGroup()
        self.create_manifold()
        self.create_points_and_metrics()
        self.add_uriem_label()
    
    def get_points_defining_boundary(self):
        """Override to return a fixed set of points."""
        return np.array([
            [-self._width/2, -self._height/2, 0],
            [self._width/2, -self._height/2, 0],
            [self._width/2, self._height/2, 0],
            [-self._width/2, self._height/2, 0]
        ], dtype=float)
    
    def get_anchors(self):
        """Override to return a fixed set of anchors."""
        return self.get_points_defining_boundary()
    
    def get_critical_point(self, direction):
        """Override to return a fixed critical point."""
        return np.array([0, 0, 0], dtype=float)
    
    def get_center(self):
        """Override get_center to return a fixed point."""
        return np.array([0, 0, 0], dtype=float)
    
    def add_uriem_label(self):
        """Add URIEM label with surrounding box."""
        uriem_label = Text("URIEM", color=COLORS["latent"]).scale(0.6)
        uriem_box = SurroundingRectangle(
            uriem_label,
            color=COLORS["latent"],
            buff=0.2,
            fill_opacity=0.1
        )
        uriem_group = VGroup(uriem_box, uriem_label)
        uriem_group.next_to(self.manifold, UP, buff=0.5)
        self.add(uriem_group)
    
    def create_manifold(self):
        """Create the latent manifold outline with URIEM emphasis."""
        progress = self.epoch / self.max_epochs if self.max_epochs > 0 else 0
        
        # Create base manifold shape as an ellipse
        self.manifold = Circle(
            radius=self._width/2,
            color=COLORS["latent"],
            fill_opacity=0.2,
            stroke_width=2
        ).stretch(self._height/self._width, 1)
        
        # Add metric tensor field visualization inside the manifold
        self.metric_field = VGroup()
        num_rings = 3
        points_per_ring = 8
        
        for r in np.linspace(0.2, 0.8, num_rings):
            for theta in np.linspace(0, 2*PI, points_per_ring, endpoint=False):
                x = r * self._width/2 * np.cos(theta)
                y = r * self._height/2 * np.sin(theta)
                
                # Create metric tensor as an ellipse
                ellipse = Ellipse(
                    width=0.4 * (1 - 0.3 * progress),
                    height=0.2 * (1 - 0.3 * progress),
                    color=COLORS["latent"],
                    fill_opacity=0.1,
                    stroke_opacity=0.5
                )
                ellipse.move_to([x, y, 0])
                ellipse.rotate(theta + PI/4)  # Rotate to follow manifold curvature
                self.metric_field.add(ellipse)
        
        self.add(self.manifold, self.metric_field)
    
    def create_points_and_metrics(self):
        """Create points and their surrounding metric tensors."""
        self.points_group = VGroup()
        self.metrics_group = VGroup()
        progress = self.epoch / self.max_epochs if self.max_epochs > 0 else 0
        
        # Sample points with more structure as training progresses
        for _ in range(self.num_points):
            # Sample angle and radius with increasing structure
            angle = np.random.uniform(0, 2*PI)
            if progress > 0.5:
                # More structured distribution later in training
                r = np.random.beta(8, 2) * self._width/3
            else:
                # More random distribution early in training
                r = np.random.uniform(0, self._width/3)
            
            # Project onto manifold with evolving noise
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            # Add noise that decreases with training
            noise_scale = 0.5 * (1 - progress)
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)
            
            # Create point
            point = Dot(
                point=[x, y, 0],
                color=COLORS["latent"],
                radius=0.05
            )
            self.points_group.add(point)
            
            # Create surrounding metric tensor ellipses
            num_ellipses = 3
            for i in range(num_ellipses):
                angle_offset = 2*PI * i / num_ellipses + progress * PI
                ellipse = Ellipse(
                    width=0.8 * (1 - 0.3 * progress),
                    height=0.4 * (1 - 0.3 * progress),
                    color=COLORS["latent"],
                    fill_opacity=0.1,
                    stroke_opacity=0.5
                )
                # Position ellipse around point
                ellipse.move_to([
                    x + 0.3 * np.cos(angle_offset),
                    y + 0.3 * np.sin(angle_offset),
                    0
                ])
                ellipse.rotate(angle_offset)
                self.metrics_group.add(ellipse)
        
        self.add(self.points_group, self.metrics_group)
    
    def highlight_sample_point(self, point_idx: int) -> Animation:
        """Create animation to highlight a specific sample point on URIEM."""
        if point_idx < len(self.points_group):
            point = self.points_group[point_idx]
            return Succession(
                point.animate.scale(2),
                point.animate.set_color(COLORS["highlight"]),
                point.animate.scale(0.5),
                point.animate.set_color(COLORS["latent"])
            )
        return Animation(self)  # Return empty animation if point doesn't exist

# PLACEMENT: Main scene class that orchestrates all components
class GUGUSTrainingPipeline(MovingCameraScene):
    """Main scene for visualizing GUGUS model training dynamics with URIEM emphasis."""
    
    def __init__(self):
        super().__init__()
        self.current_mobjects = VGroup()
        self.num_epochs = 5
        self.num_points = 5
        self.patient_data = self.generate_patient_data()
    
    def generate_patient_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic patient trajectory data."""
        data = {}
        points = []
        
        # Generate exactly 5 points along a curve with strategic placement
        strategic_positions = [
            (0.1*PI, 0.98),    # Right vertex (past)
            (0.5*PI, 1.8*0.98),  # Upper right vertex (past)
            (1.0*PI, 0.8*0.98),  # Top vertex (present)
            (1.2*PI, 2*0.98),    # Upper left vertex (future)
            (1.6*PI, 0.98),    # Left vertex (future)
        ]
        
        for angle, radius_factor in strategic_positions:
            # Add very small randomness to keep points at vertices
            perturbed_angle = angle + np.random.uniform(-0.05, 0.05)
            base_point = blob_manifold(perturbed_angle)
            point = radius_factor * np.array([base_point[0], base_point[1], 0])
            points.append(point)
        
        data["patient_0"] = np.array(points)
        return data
    
    def create_trajectory(self, points: np.ndarray, color: str = COLORS["patient"]) -> VGroup:
        """Create a trajectory visualization from points."""
        trajectory = VGroup()
        dots = VGroup()
        labels = VGroup()
        arrows = VGroup()
        
        # Define colors for past and future points
        past_color = "#FF6B6B"  # Light red for past
        future_color = "#4ECDC4"  # Turquoise for future
        present_color = COLORS["highlight"]  # Yellow for present
        
        # Ensure we have exactly 5 points
        if len(points) > 5:
            points = points[:5]
        
        for i, point in enumerate(points):
            # Choose color based on position
            if i < 2:  # Past points
                point_color = past_color
            elif i == 2:  # Present point
                point_color = present_color
            else:  # Future points
                point_color = future_color
            
            # Create dot
            dot = Dot(
                point=point,
                color=point_color,
                radius=0.08 if i == 2 else 0.06  # Slightly larger for present point
            )
            
            # Create label
            label = MathTex(
                f"x_{{i,{i+1}}}",
                color=point_color
            ).scale(0.6)
            label.next_to(dot, UP + RIGHT, buff=0.1)
            
            dots.add(dot)
            labels.add(label)
            
            # Create arrow to next point
            if i < len(points) - 1:
                arrow = Arrow(
                    start=dot.get_center(),
                    end=points[i + 1],
                    color=point_color,
                    buff=0.2,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )
                arrows.add(arrow)
        
        trajectory.add(dots, labels, arrows)
        return trajectory
    
    def show_training_step(self):
        """Show a single training step with URIEM emphasis."""
        # Set initial camera position
        self.camera.frame.set_width(35)
        self.camera.frame.move_to(ORIGIN + UP * 1)
        
        # Create title with URIEM emphasis
        title = Text("GUGUS Training on URIEM", color=COLORS["text"])
        title.scale(1.8).to_edge(4 * UP, buff=0.8)
        title.move_to(UP * 8)
        
        # Create elements with URIEM emphasis
        # Original distribution (left)
        manifold = ParametricFunction(
            blob_manifold,
            t_range=[0, 2*PI],
            color=COLORS["patient"],
            fill_opacity=0.3
        ).scale(2.5)
        manifold.move_to(LEFT * 13)
        
        # Create trajectory
        data = self.patient_data["patient_0"]
        trajectory = self.create_trajectory(data)
        trajectory.move_to(manifold.get_center())
        
        trajectory = trajectory[:len(trajectory)//2]
        # Add numbered labels for trajectory points
        trajectory_labels = VGroup()
        for i, dot in enumerate(trajectory[0]):
            label = MathTex(f"x_{{{i+1}}}", color=COLORS["patient"]).scale(0.6)
            label.next_to(dot, UP + RIGHT, buff=0.2)
            trajectory_labels.add(label)
        
        # Create encoder network
        encoder = EncoderNetwork(input_dim=2, hidden_dims=[4, 8, 4])
        encoder.scale(0.6)
        encoder.move_to(manifold.get_center() + RIGHT * 6)
        
        # Add encoder label with URIEM emphasis
        encoder_label = MathTex("q_\\phi(z|x) \\text{ on URIEM}", color=COLORS["encoder"]).scale(0.8)
        encoder_label.next_to(encoder, DOWN, buff=0.4)
        
        # Define trajectory evolution function for reconstruction
        def get_evolved_trajectory(progress):
            # Calculate noise scale that decreases with training progress
            noise_scale = 0.5 * (1 - progress)
            evolved_data = []
            for point in data:
                # Add 3D noise (x,y only, preserving z)
                noise = np.array([
                    np.random.normal(0, noise_scale),
                    np.random.normal(0, noise_scale),
                    0  # Keep z-coordinate unchanged
                ])
                evolved_point = point + noise
                evolved_data.append(evolved_point)
            return np.array(evolved_data)
        
        # Create latent space with URIEM emphasis
        latent_space = LatentSpace(width=8.0, height=8.0)
        latent_space.move_to(encoder.get_center() + RIGHT * 7)
        
        # Create diffusion process inside latent space
        diffusion = DiffusionProcess(
            start_point=latent_space.get_center(),
            num_steps=5
        )
        # Position diffusion in bottom part of latent space
        diffusion.move_to(latent_space.get_center() + DOWN * 0.1)
        
        # Create initial reconstructed trajectory
        reconstructed_trajectory = self.create_trajectory(data)
        reconstructed_trajectory.move_to(manifold.get_center())
        
        # Create decoder network
        decoder = DecoderNetwork(input_dim=2, hidden_dims=[4, 8, 4])
        decoder.scale(0.6)
        decoder.move_to(latent_space.get_center() + RIGHT * 8)
        
        # Add decoder label
        decoder_label = MathTex("p_\\theta(x|z)", color=COLORS["decoder"]).scale(0.8)
        decoder_label.next_to(decoder, DOWN, buff=0.4)
        
        # Create reconstructed distribution
        reconstructed_manifold = ParametricFunction(
            blob_manifold,
            t_range=[0, 2*PI],
            color=COLORS["patient"],
            fill_opacity=0.3,
            stroke_opacity=1.0
        ).scale(2.5)
        reconstructed_manifold.move_to(decoder.get_center() + RIGHT * 7)
        
        # Add "Reconstruction" label
        recon_label = Text("Reconstruction", color=COLORS["patient"]).scale(0.6)
        recon_label.next_to(reconstructed_manifold, UP, buff=0.4)
        
        # Create loss visualization with URIEM emphasis
        loss_viz = LossVisualization(max_width=4.0)
        loss_viz.move_to(diffusion.get_center() + DOWN * 6)
        
        # Show initial state
        self.play(
            Write(title),
            Create(manifold),
            Write(trajectory_labels),
            run_time=1
        )
        
        # Show trajectory
        self.play(
            Create(trajectory),
            run_time=1.5
        )
        
        # Show encoder with URIEM emphasis
        self.play(
            Create(encoder),
            Write(encoder_label),
            run_time=1.5
        )
        
        # Show latent space with URIEM emphasis
        self.play(
            Create(latent_space),
            #Create(diffusion),
            run_time=1.5
        )
        
        # Show decoder and reconstruction
        self.play(
            Create(decoder),
            Write(decoder_label),
            Create(reconstructed_manifold),
            Write(recon_label),
            run_time=1.5
        )
        
        # Show loss visualization
        self.play(
            Create(loss_viz),
            run_time=1.5
        )
        
        # Training animation
        first_diffusion = None  # Keep track of the first diffusion
        
        for epoch in range(self.num_epochs):
            progress = epoch / (self.num_epochs - 1)
            
            # Update latent space
            new_latent_space = LatentSpace(
                width=8.0,
                height=8.0,
                epoch=epoch,
                max_epochs=self.num_epochs
            )
            new_latent_space.move_to(latent_space.get_center())
            
            # Update diffusion process
            new_diffusion = DiffusionProcess(
                start_point=new_latent_space.get_center(),
                num_steps=5
            )
            new_diffusion.move_to(new_latent_space.get_center() + DOWN * 4)
            
            # Update reconstructed manifold and trajectory
            evolved_data = get_evolved_trajectory(progress)
            new_reconstructed_manifold = ParametricFunction(
                blob_manifold,
                t_range=[0, 2*PI],
                color=COLORS["patient"],
                fill_opacity=0.3 + 0.4 * progress,
                stroke_opacity=1.0
            ).scale(2.5 * (0.8 + 0.4 * progress))
            new_reconstructed_manifold.move_to(reconstructed_manifold.get_center())
            
            # Create evolved trajectory
            new_reconstructed_trajectory = self.create_trajectory(evolved_data)
            new_reconstructed_trajectory.move_to(new_reconstructed_manifold.get_center())
            
            # Update loss visualization
            loss_viz.update_values(progress)
            
            # Animate transitions with trajectory
            if epoch == 0:
                # First show the yellow center point
                self.play(
                    Create(new_diffusion.points_group[0]),  # Center point
                    run_time=0.5
                )
                # Then show the forward diffusion
                self.play(
                    Create(new_diffusion.points_group[1:]),  # Remaining points
                    Create(new_diffusion.forward_arrows),
                    Create(new_diffusion.noise_circles),
                    run_time=1.0
                )
                first_diffusion = new_diffusion  # Store reference to first diffusion
                diffusion = new_diffusion
            elif epoch == 1:
                # Remove the first diffusion completely
                if first_diffusion:
                    self.play(
                        *[FadeOut(mob) for mob in [
                            first_diffusion.points_group,
                            first_diffusion.forward_arrows,
                            first_diffusion.noise_circles,
                            *first_diffusion.formulas
                        ]],
                        run_time=0.5
                    )
                self.play(
                    Transform(latent_space, new_latent_space),
                    Create(new_diffusion),
                    Transform(reconstructed_manifold, new_reconstructed_manifold),
                    Transform(reconstructed_trajectory, new_reconstructed_trajectory),
                    run_time=1.0
                )
                diffusion = new_diffusion
            else:
                # Remove everything from old diffusion
                self.play(
                    *[FadeOut(mob) for mob in [
                        diffusion.points_group,
                        diffusion.forward_arrows,
                        diffusion.noise_circles,
                        *diffusion.formulas
                    ]],
                    run_time=0.5
                )
                self.play(
                    Transform(latent_space, new_latent_space),
                    Create(new_diffusion),
                    Transform(reconstructed_manifold, new_reconstructed_manifold),
                    Transform(reconstructed_trajectory, new_reconstructed_trajectory),
                    run_time=1.0
                )
                diffusion = new_diffusion
        
        # Final state
        final_text = Text("Training Complete on URIEM", color=COLORS["highlight"]).scale(0.7)
        final_text.next_to(title, DOWN, buff=0.4)
        self.play(Write(final_text))
        
        self.wait(2)
    
    def construct(self):
        """Main construction sequence."""
        self.show_training_step()
        self.wait(2) 