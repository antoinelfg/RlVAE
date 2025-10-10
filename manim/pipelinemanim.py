from manim import *
from manim_slides import Slide
from manim import Paragraph
import numpy as np

# Configuration settings
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 30
config.media_width = "100%"
config.background_color = BLACK
config.frame_height = 8.0
config.frame_width = 14.222222222222221
config.output_file = "KMeansMetricVisualization"
config.format = "mp4"
config.disable_caching = True
config.flush_cache = True

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

# PLACEMENT: Top center of the scene
class TBlockScheme(VGroup):
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
class TRiemannianDistribution(VGroup):
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
class TDiffusionProcess(VMobject):
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
class TEncoderNetwork(VGroup):
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
class TRiemannianMetricField(VGroup):
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
class TLossVisualization(VGroup):
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
class TDecoderNetwork(VGroup):
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
class TLatentSpace(VMobject):
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

class ALL(MovingCameraScene, Slide):
    def flash_animation(self, mobject):
        return Succession(
            mobject.animate.set_color(YELLOW),
            mobject.animate.set_color(mobject.get_color()),
        )

    def construct(self):
        # SLIDE 1: Title with dynamic entrance
        title = Text("Vanilla VAE — Training Phase", font_size=36)
        subtitle = Text("Understanding the Architecture", font_size=24, color=BLUE)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP, run_time=1)
        )
        self.next_slide()

        # SLIDE 2: Full Pipeline with dynamic build-up
        # Initialize components with initial opacity 0
        x = MathTex(r"x_j^{(i)}", color=YELLOW).scale(1.2).move_to(LEFT * 6.5)
        encoder = Rectangle(width=3, height=1.5, color=BLUE).move_to(LEFT * 4)
        enc_lbl = Text("Encoder", font_size=24, color=BLUE).next_to(encoder, UP, buff=0.1)
        enc1 = MathTex(r"q_{\phi}(z|x)", color=BLUE_C).scale(0.6).move_to(encoder.get_center() + UP * 0.3)
        enc2 = MathTex(r"\mathcal N(\mu_{\phi}(x_j^{(i)}), \sigma_{\phi}(x_j^{(i)}))", color=BLUE_C).scale(0.5).next_to(enc1, DOWN, buff=0.2)
        
        # Create latent space with gradient fill
        latent = RoundedRectangle(corner_radius=0.2, width=3, height=3, color=ORANGE)
        latent_fill = latent.copy().set_fill(color=ORANGE, opacity=0.1)
        latent_group = VGroup(latent, latent_fill).move_to(ORIGIN)
        lat_lbl = Text("Latent Space", font_size=24, color=ORANGE).next_to(latent, UP, buff=0.1)
        
        decoder = Rectangle(width=3, height=1.5, color=GREEN).move_to(RIGHT * 4)
        dec_lbl = Text("Decoder", font_size=24, color=GREEN).next_to(decoder, UP, buff=0.1)
        dec1 = MathTex(r"p_{\theta}(z|x)", color=GREEN_C).scale(0.6).move_to(decoder.get_center() + UP * 0.3)
        dec2 = MathTex(r"\mathcal N(\mu_{\theta}(z_j^{(i)}), diag(\sigma_{\theta}^2(z_j^{(i)})))", color=GREEN_C).scale(0.45).next_to(dec1, DOWN, buff=0.2)
        xhat = MathTex(r"\hat x_j^{(i)}", color=YELLOW).scale(1.2).move_to(RIGHT * 6.5)

        # Create arrows with gradient
        arr1 = Arrow(x.get_right(), encoder.get_left(), buff=0.1, color=BLUE)
        arr2 = Arrow(encoder.get_right(), latent.get_left(), buff=0.1, color=ORANGE)
        arr3 = Arrow(latent.get_right(), decoder.get_left(), buff=0.1, color=GREEN)
        arr4 = Arrow(decoder.get_right(), xhat.get_left(), buff=0.1, color=YELLOW)

        # Prior information with dynamic appearance
        prior_lbl = Text("Prior:", font_size=24).move_to(latent.get_center())
        zij = MathTex(r"z_j^{(i)}", color=ORANGE).scale(0.9).next_to(prior_lbl, UP, buff=0.2)
        prior = MathTex(r"p(z)=\mathcal N(0,I)", color=ORANGE).scale(0.8).next_to(prior_lbl, DOWN, buff=0.2)

        # Animate components sequentially with effects
        self.play(FadeOut(title_group, shift=UP))
        
        # Input data animation
        self.play(
            Write(x),
            Create(VGroup(encoder, enc_lbl)),
            run_time=0.5
        )
        
        # Encoder equations with typewriter effect
        self.play(
            AddTextLetterByLetter(enc1),
            AddTextLetterByLetter(enc2),
            run_time=0.5
        )
        
        # Arrow animation with trace effect
        self.play(
            ShowPassingFlash(arr1.copy().set_color(WHITE), time_width=0.5),
            Create(arr1)
        )

        # Latent space appearance with growing effect
        self.play(
            Create(latent),
            FadeIn(latent_fill),
            Write(lat_lbl),
            run_time=0.5
        )
        
        # Prior information with fade effect
        self.play(
            Write(prior_lbl),
            FadeIn(zij, shift=DOWN),
            FadeIn(prior, shift=UP),
            run_time=0.5
        )

        # Second arrow with pulse
        self.play(
            ShowPassingFlash(arr2.copy().set_color(WHITE), time_width=0.5),
            Create(arr2)
        )

        # Decoder appearance with build-up
        self.play(
            Create(VGroup(decoder, dec_lbl)),
            run_time=0.5
        )
        
        # Decoder equations with typewriter effect
        self.play(
            AddTextLetterByLetter(dec1),
            AddTextLetterByLetter(dec2),
            run_time=0.5
        )

        # Final arrows and output
        self.play(
            ShowPassingFlash(arr3.copy().set_color(WHITE), time_width=0.5),
            Create(arr3)
        )
        self.play(
            ShowPassingFlash(arr4.copy().set_color(WHITE), time_width=0.5),
            Create(arr4),
            Write(xhat)
        )

        self.next_slide()

        # SLIDE 3: Highlight Latent Jacobian with dynamic effects
        latgroup = VGroup(latent, lat_lbl)
        extraction_box1 = SurroundingRectangle(latgroup, color=RED, buff=0.3)
        jac_arrow1 = Arrow(extraction_box1.get_bottom(), extraction_box1.get_bottom() + DOWN * 1.0, color=RED)
        jac_label1 = MathTex(r"c_k \;=\;\frac{1}{|\mathcal{I}k|}\sum_{(i,j)\in\mathcal{I}_k} z_j^{(i)}", color=RED).scale(0.7)
        
        # Add glow effect to the box
        glow = extraction_box1.copy().set_stroke(color=RED, opacity=0.5, width=10)
        
        # Animate highlight with glow
        self.play(
            Create(extraction_box1),
            FadeIn(glow, rate_func=there_and_back),
            run_time=0.5
        )
        
        # Animate arrow and label with dynamic effect
        self.play(
            Create(jac_arrow1),
            Write(jac_label1.next_to(jac_arrow1, DOWN, buff=0.1)),
            run_time=0.5
        )

        centro = Text("Centroids via k-means:", font_size=24, color=RED)
        self.play(
            Write(centro.next_to(extraction_box1, UP, buff=0.1)),
            jac_label1.animate.set_color(RED),
            run_time=0.5
        )

        # Decoder highlight with similar effects
        decgroup = VGroup(decoder, dec_lbl)
        extraction_box2 = SurroundingRectangle(decgroup, color=YELLOW, buff=0.3)
        glow2 = extraction_box2.copy().set_stroke(color=YELLOW, opacity=0.5, width=10)
        
        self.play(
            Create(extraction_box2),
            FadeIn(glow2, rate_func=there_and_back),
            run_time=0.5
        )

        jac_arrow2 = Arrow(extraction_box2.get_bottom(), extraction_box2.get_bottom() + DOWN * 1.0, color=YELLOW)
        jac_label2 = MathTex(r"J_\theta(z)", color=YELLOW).scale(0.7)
        
        self.play(
            Create(jac_arrow2),
            Write(jac_label2.next_to(jac_arrow2, DOWN, buff=0.1)),
            run_time=0.5
        )

        jaco = Text("Access to Jacobians:", font_size=24, color=YELLOW)
        self.play(
            Write(jaco.next_to(extraction_box2, UP, buff=0.1)),
            run_time=0.5
        )

        # Final equation with dynamic build-up
        eqm = MathTex(r"M_k =J_\theta(c_k)^{\!T}\,J_\theta(c_k)", color=YELLOW).scale(0.7)
        self.play(
            Write(eqm.next_to(jac_label2, DOWN, buff=0.1)),
            run_time=0.5
        )

        self.next_slide()

        # SLIDE 5: ELBO
        #elbo = MathTex(
        #    r"\mathcal L(x)=\mathbb E_{q(z|x)}[\log p(x|z)]"
        #    r"-\mathrm{KL}(q\|p)"
        #).scale(0.8).to_edge(DOWN)
        #self.remove(prior, kl)
        #self.play(FadeIn(elbo)) 

        # SLIDE 6: Metric Extraction
         # Clean up VAE components with fade out effect
        vae_components = VGroup(
            x, encoder, enc_lbl, enc1, enc2,
            latent, latent_fill, lat_lbl, zij,
            decoder, dec_lbl, dec1, dec2,
            xhat, arr1, arr2, arr3, arr4,
            prior_lbl, prior,
            extraction_box1, jac_arrow1, jac_label1,
            extraction_box2, jac_arrow2, jac_label2,
            centro, jaco, eqm,
            glow, glow2
        )
        
        self.play(
            *[FadeOut(obj, shift=DOWN * 0.5) for obj in vae_components],
            run_time=0.5
        )
        
        met_title = Text("Metric Extraction", font_size=36).to_edge(UP)
        self.play(Write(met_title), run_time=1)

        # Centroid equation with dynamic build-up
        jac_label1 = MathTex(r"c_k \;=\;\frac{1}{|\mathcal{I}k|}\sum_{(i,j)\in\mathcal{I}_k} z_j^{(i)}", color=RED).scale(0.7)
        jac_label1.move_to(UP * 2 + LEFT * 5)
        self.play(Write(jac_label1), run_time=0.5)

        # Precision matrix equation with highlight
        eqm = MathTex(r"M_k =J_\theta(c_k)^{\!T}\,J_\theta(c_k)", color=YELLOW).scale(0.7)
        eqm.next_to(jac_label1, DOWN, buff=0.5)
        self.play(Write(eqm), run_time=0.5)

        # Add inverse covariance with flash effect
        eqm2 = MathTex(r" =\Sigma_k^{-1}\; }", color=YELLOW).scale(0.7)
        eqm2.next_to(eqm, RIGHT, buff=0.1)
        self.play(
            Write(eqm2),
            Flash(eqm2, color=YELLOW, line_length=0.2),
            run_time=0.5
        )
        eqmgroup = VGroup(eqm, eqm2)

        # Weight equation with sequential animation
        wk = MathTex(
            r"w_k(z_{j}^{(i)})\;=\;\frac{e^{-\frac{\|z_{j}^{(i)} - c_k\|^2}{2\,\lambda\,T}}}"
            r"{\displaystyle \sum_{\ell=1}^{K} e^{-\frac{\|z_{j}^{(i)} - c_{\ell}\|^2}{2\,\lambda\,T}}}",
            color=ORANGE
        ).scale(0.7)
        wk.next_to(eqm, DOWN, buff=0.5)
        self.play(Write(wk), run_time=1)

        # Arrange equations
        all_group = VGroup(jac_label1, eqmgroup, wk).arrange(
            DOWN, 
            aligned_edge=LEFT, 
            buff=0.5
        ).move_to(LEFT * 4 + UP * 0.5)

        # Add explanations with dynamic arrows
        centroids = Text("Centroids obtained via k-means", font_size=24, color=WHITE)
        centroids.next_to(jac_label1, RIGHT * 5, buff=0.5)
        arr6 = Arrow(jac_label1.get_right(), centroids.get_left(), buff=0.2)
        self.play(
            Create(arr6),
            Write(centroids),
            run_time=0.5
        )

        mk2 = Paragraph(
            "The precision matrix (the inverse covariance) of \nlocal latent fluctuations",
            font_size=24,
            color=WHITE,
            line_spacing=0.3,
            width=6
        ).next_to(eqmgroup, RIGHT * 4, buff=0.5)
        arr7 = Arrow(eqmgroup.get_right(), mk2.get_left(), buff=0.2)
        self.play(
            Create(arr7),
            AddTextLetterByLetter(mk2),
            run_time=0.5
        )

        wk2 = Paragraph(
            "The weights of the local latent fluctuations",
            font_size=24,
            color=WHITE,
            line_spacing=0.3,
            width=6
        ).next_to(wk, RIGHT * 4, buff=0.5)
        arr8 = Arrow(wk.get_right(), wk2.get_left(), buff=0.2)
        self.play(
            Create(arr8),
            AddTextLetterByLetter(wk2),
            run_time=0.5
        )

        # Parameters with highlighting
        lambda_def = MathTex(
            r"\lambda > 0:",
            r"\text{ kernel bandwidth}",
            tex_to_color_map={r"\lambda": ORANGE, "bandwidth": ORANGE}
        ).scale(0.5)
        lambda_def.next_to(wk, DOWN, buff=0.4).align_to(wk, LEFT)
        self.play(
            Write(lambda_def),
            Flash(lambda_def[0], color=ORANGE, line_length=0.2),
            run_time=0.5
        )

        T_def = MathTex(
            r"T > 0:",
            r"\text{ softmax temperature}",
            tex_to_color_map={r"T": ORANGE, "temperature": ORANGE}
        ).scale(0.5)
        T_def.next_to(lambda_def, DOWN, buff=0.2).align_to(lambda_def, LEFT)
        self.play(
            Write(T_def),
            Flash(T_def[0], color=ORANGE, line_length=0.2),
            run_time=0.5
        )

        # Final equation with box animation
        eq = MathTex(
            r"\boxed{"
            r"G^{-1}(z_{j}^{(i)}) \;=\; \sum_{k=1}^K w_k(z_{j}^{(i)}) \, M_k \;\Longrightarrow\;G(z_{j}^{(i)})\approx\bigl(G^{-1}(z_{j}^{(i)})+\varepsilon I\bigr)^{-1}\!"
            r"}",
            tex_to_color_map={
                r"G^{-1}(z_{i}^j)": WHITE,
                r"G(z_{i}^j)": WHITE,
                r"w_k(z_{i}^j)": WHITE,
                r"M_k": WHITE,
                r"\Longrightarrow": WHITE,
                r"\sum": WHITE,
            }
        ).scale(0.6).move_to(RIGHT * 2.4 + DOWN * 2.6)
        
        box = SurroundingRectangle(eq, buff=0.2)
        self.play(Write(eq), run_time=1)
        self.play(Create(box), run_time=0.5)
        self.play(FadeOut(box), run_time=0.5)

        self.next_slide()
        self.wait(2)

        # Clean up everything
        final_components = VGroup(
            met_title, all_group,
            centroids, arr6, mk2, arr7, wk2, arr8,
            lambda_def, T_def, eq
        )
        
        self.play(
            *[FadeOut(obj, shift=DOWN * 0.5) for obj in final_components],
            run_time=0.5
        )

        self.next_slide()

        # Start VAE Visualization section
        title = Text("Vanilla VAE: Encoding and Decoding", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create three sections for input, latent, and output spaces
        input_frame = Rectangle(height=4, width=3, color=YELLOW)
        input_frame.move_to(LEFT * 4)
        latent_frame = Rectangle(height=4, width=3, color=RED)
        latent_frame.move_to(ORIGIN)
        output_frame = Rectangle(height=4, width=3, color=ORANGE)
        output_frame.move_to(RIGHT * 4)

        # Labels for each space
        input_label = MathTex(r"x_j \sim \text{Data}", color=YELLOW).next_to(input_frame, UP)
        latent_label = MathTex(r"z_j \sim q_\phi(z|x)", color=RED).next_to(latent_frame, UP)
        output_label = MathTex(r"\hat{x}_j \sim p_\theta(x|z)", color=ORANGE).next_to(output_frame, UP)

        self.play(
            Create(input_frame),
            Create(latent_frame),
            Create(output_frame),
            Write(input_label),
            Write(latent_label),
            Write(output_label)
        )
        self.next_slide()

        # Create manifold-like input figure (swiss roll-like shape)
        t = np.linspace(0, 4*np.pi, 100)
        input_points = []
        for theta in t:
            # Create a curved manifold shape
            r = 0.3 * (1 + 0.2 * np.cos(3*theta))
            x = r * np.cos(theta)
            y = r * np.sin(theta) + 0.1 * np.cos(5*theta)
            input_points.append([x, y, 0])

        # Add some perpendicular variation to create a 2D manifold
        manifold_points = []
        for p in input_points:
            # Add points perpendicular to the curve to create width
            for w in np.linspace(-0.15, 0.15, 5):
                # Calculate tangent vector
                dx = -p[1]
                dy = p[0]
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    dx, dy = dx/norm, dy/norm
                    manifold_points.append([
                        p[0] + w*dx,
                        p[1] + w*dy,
                        0
                    ])

        input_shape = VMobject(color=YELLOW, fill_opacity=0.2)
        input_shape.set_points_smoothly([
            input_frame.get_center() + np.array(p) for p in manifold_points[::5]
        ])
        
        # Add some points on the manifold
        input_dots = VGroup(*[
            Dot(input_frame.get_center() + np.array(p), color=YELLOW_E, radius=0.05)
            for p in manifold_points[::25]
        ])
        
        self.play(
            Create(input_shape),
            LaggedStart(*[FadeIn(dot) for dot in input_dots], lag_ratio=0.1)
        )

        # Create latent space visualization
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            axis_config={"stroke_color": GRAY, "stroke_width": 1},
        ).scale(0.7)
        axes.move_to(latent_frame.get_center())
        
        # Prior distribution
        prior_circle = Circle(radius=1.2, color=RED, fill_opacity=0.1)
        prior_circle.move_to(axes.get_center())
        prior_label = MathTex(r"p(z) = \mathcal{N}(0, I)", color=RED).scale(0.6)
        prior_label.next_to(prior_circle, UP, buff=0.1)

        self.play(
            Create(axes),
            Create(prior_circle),
            Write(prior_label)
        )

        # Encode points into latent space with noise clouds (constrained within prior)
        encoded_points = []
        for dot in input_dots:
            # Get relative position and scale to fit within prior
            pos = dot.get_center() - input_frame.get_center()
            # Normalize to ensure points stay within prior radius
            norm = np.sqrt(pos[0]**2 + pos[1]**2)
            if norm > 0:
                scale = min(1.0, prior_circle.radius / norm)
                encoded_pos = axes.c2p(pos[0]*scale, pos[1]*scale)
                encoded_points.append(encoded_pos)

        encoded_clouds = VGroup(*[
            Circle(radius=0.2, color=BLUE, fill_opacity=0.15).move_to(pos)
            for pos in encoded_points
        ])

        encoded_dots = VGroup(*[
            Dot(pos, color=BLUE)
            for pos in encoded_points
        ])

        # Add particles around encoded points (constrained within prior)
        for cloud in encoded_clouds:
            cloud_center = cloud.get_center()
            particles = VGroup(*[
                Dot(
                    point=cloud_center + np.array([
                        np.random.normal(0, 0.1),
                        np.random.normal(0, 0.1),
                        0
                    ]) * 0.8,  # Scale factor to keep particles closer to center
                    radius=0.02,
                    color=BLUE_A
                )
                for _ in range(10)
            ])
            # Ensure particles stay within prior
            for particle in particles:
                pos = particle.get_center() - axes.get_center()
                norm = np.sqrt(pos[0]**2 + pos[1]**2)
                if norm > prior_circle.radius:
                    scale = prior_circle.radius / norm
                    particle.move_to(axes.get_center() + pos * scale)
            cloud.particles = particles

        # Animate encoding
        encode_arrow = Arrow(input_frame.get_right(), latent_frame.get_left(), color=BLUE, buff=0.2)
        encode_text = MathTex(r"q_\phi(z|x)", color=BLUE).next_to(encode_arrow, UP, buff=0.1).scale(0.5)

        self.play(
            Create(encode_arrow),
            Write(encode_text)
        )

        for cloud, dot in zip(encoded_clouds, encoded_dots):
            self.play(
                Create(cloud),
                FadeIn(dot),
                LaggedStart(*[FadeIn(p) for p in cloud.particles], lag_ratio=0.05),
                run_time=0.5
            )

        # Create output manifold (slightly distorted version of input)
        output_points = []
        for p in manifold_points:
            noise = np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05), 0])
            output_points.append(np.array(p) + noise)

        output_shape = VMobject(color=ORANGE, fill_opacity=0.2)
        output_shape.set_points_smoothly([
            output_frame.get_center() + np.array(p) for p in output_points[::5]
        ])
        
        output_dots = VGroup(*[
            Dot(output_frame.get_center() + np.array(p), color=ORANGE, radius=0.05)
            for p in output_points[::25]
        ])

        # Animate decoding
        decode_arrow = Arrow(latent_frame.get_right(), output_frame.get_left(), color=GREEN, buff=0.2)
        decode_text = MathTex(r"p_\theta(x|z)", color=GREEN).next_to(decode_arrow, UP, buff=0.1).scale(0.5)

        self.play(
            Create(decode_arrow),
            Write(decode_text)
        )

        self.play(
            Create(output_shape),
            LaggedStart(*[FadeIn(dot) for dot in output_dots], lag_ratio=0.1)
        )

        # Add loss terms
        kl_div = MathTex(r"\text{KL}(q_\phi(z|x) \| p(z))", color=WHITE).scale(0.7)
        kl_div.next_to(latent_frame, DOWN, buff=0.2)
        
        rec_loss = MathTex(r"\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]", color=WHITE).scale(0.7)
        rec_loss.next_to(kl_div, DOWN, buff=0.2)

        self.play(Write(kl_div))
        self.play(Write(rec_loss))

        self.next_slide()
        self.wait(2)

        # Clean up
        vae_components = VGroup(
            title, input_frame, latent_frame, output_frame,
            input_label, latent_label, output_label,
            input_shape, input_dots,
            axes, prior_circle, prior_label,
            encoded_clouds, encoded_dots,
            encode_arrow, encode_text,
            output_shape, output_dots,
            decode_arrow, decode_text,
            kl_div, rec_loss
        )
        
        for cloud in encoded_clouds:
            vae_components.add(cloud.particles)

        self.play(
            *[FadeOut(obj, shift=DOWN * 0.5) for obj in vae_components],
            run_time=1.5
        )

        self.next_slide()

        # Start Diffusion section
        title = Text("Latent Diffusion: Pre-training", font_size=42).to_edge(UP)
        self.play(Write(title))
        self.next_slide()
        
        # Prompt line
        embed_prompt = MathTex(
            r"\text{1 - For each sequence }(i), \text{ encode only its final observation:}"
        ).scale(0.8).next_to(title, DOWN, buff=0.6)

        # Equation line, broken into two parts for readability
        embed_eq = MathTex(
            r"z_{T_i}^{(i)} \sim q_{\phi}\bigl(z \mid x_{T_i}^{(i)}\bigr)",
            r"= \mathcal{N}\bigl(\mu_{\phi}(x_{T_i}^{(i)}),\,\mathrm{diag}\bigl(\sigma_{\phi}^2(x_{T_i}^{(i)})\bigr)\bigr)",
            tex_to_color_map={
                r"z_{T_i}^{(i)}": YELLOW,
                r"q_{\phi}": BLUE,
                r"\mu_{\phi}": GREEN,
                r"\sigma_{\phi}": RED
            }
        ).scale(0.8).next_to(embed_prompt, DOWN, buff=0.4)
        diffusiongroup1 = VGroup(embed_prompt, embed_eq)
        embed_box = SurroundingRectangle(diffusiongroup1, color=RED, buff=0.3)

        self.play(Write(embed_prompt), Write(embed_eq), Write(embed_box))
        self.next_slide()

        # Forward noising (diffusion) prompt
        diffusion_prompt = MathTex(
            r"\text{2 - Forward diffusion (noising) at step } t:",
        ).scale(0.8).next_to(embed_eq, DOWN, buff=0.8)
        diffusion_eq = MathTex(
            r"z_t^{(i)} = \sqrt{\bar{\alpha}_t} z_{T_i}^{(i)} + \sqrt{1 - \bar{\alpha}_t} \varepsilon",
            r", \quad \varepsilon \sim \mathcal{N}(0,I)",
            tex_to_color_map={
                r"\sqrt{\bar{\alpha}_t}": BLUE,
                r"\sqrt{1 - \bar{\alpha}_t}": BLUE,
                r"\varepsilon": GREEN,
                r"\mathcal{N}(0,I)": GREEN
            }
        ).scale(0.7).next_to(diffusion_prompt, DOWN, buff=0.4)
        diffusiongroup2 = VGroup(diffusion_prompt, diffusion_eq)
        diffusion_box = SurroundingRectangle(diffusiongroup2, color=BLUE, buff=0.3)


        # Forward noising (diffusion) prompt
        diffusion_prompt2 = MathTex(
            r"\text{3 - Reverse diffusion (denoising) at step } t:",
        ).scale(0.8).next_to(diffusion_eq, DOWN, buff=0.8)
        diffusion_eq2 = MathTex(
            r"z_{t-1}^{(i)} = \frac{1}{\sqrt{\alpha_t}} \left(z_t^{(i)} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(z_t^{(i)}, t) \right) + \sigma_t^{(i)} \, u",
            r", \quad u \sim \mathcal{N}(0,I)",
            tex_to_color_map={
                r"\sqrt{\bar{\alpha}_t}": BLUE,
                r"\sqrt{1 - \bar{\alpha}_t}": BLUE,
                r"\varepsilon": GREEN,
                r"\mathcal{N}(0,I)": GREEN
            }
        ).scale(0.7).next_to(diffusion_prompt2, DOWN, buff=0.4)
        diffusiongroup3 = VGroup(diffusion_prompt2, diffusion_eq2)
        diffusion_box2 = SurroundingRectangle(diffusiongroup3, color=GREEN, buff=0.3)
        # Play everything in one go
        self.play(
            Write(diffusion_prompt),
            Write(diffusion_eq),
            run_time=2
        )
        self.play(Write(diffusion_box))
        self.next_slide()

        self.play(
            Write(diffusion_prompt2),
            Write(diffusion_eq2),
            run_time=2
        )
        self.play(Write(diffusion_box2))
        self.next_slide()

        self.remove(embed_prompt, embed_eq, diffusion_prompt, diffusion_eq, diffusion_prompt2, diffusion_eq2)
        self.remove(embed_box, diffusion_box, diffusion_box2)
        self.remove(title)

    

        title = Text("Real Latent Diffusion Process", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Axes setup
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"stroke_color": GRAY, "stroke_width": 1},
        ).scale(0.9)
        axes.to_corner(LEFT + DOWN, buff=0.7)
        self.play(Create(axes))
        self.next_slide()

        # Time axis
        T = 4
        time_line = NumberLine(
            x_range=[1, T, 1],
            length=axes.width,
            include_numbers=True,
            tick_size=0.1,
        ).next_to(axes, DOWN, buff=1.2)
        t_label = Text("t", font_size=24).next_to(time_line, LEFT, buff=0.2)
        pointer = Dot(time_line.n2p(T), color=YELLOW)
        self.play(Create(time_line), Write(t_label), FadeIn(pointer))
        self.next_slide()

        # Forward diffusion (noising) z_T to z_1
        forward_positions = [
            axes.c2p(0.0, 0.0),
            axes.c2p(-0.3, 0.4),
            axes.c2p(-0.4, 0.0),
            axes.c2p(-0.7, 0.4),
            axes.c2p(-1.2, -0.7),
        ]

        forward_positions.reverse()

        # Create point labels with fade-in effect
        z_labels = []
        for i, pos in enumerate(forward_positions):
            if i == 0:
                label = MathTex("z_T", color=BLUE).scale(0.5)
            elif i == len(forward_positions) - 1:
                label = MathTex("z_1", color=BLUE).scale(0.5)
            else:
                label = MathTex(f"z_{len(forward_positions)-i}", color=BLUE).scale(0.5)
            label.next_to(pos, UP)
            z_labels.append(label)

        # Initialize forward trail segments with pulsing dot
        fwd_dot = Dot(forward_positions[0], color=BLUE)
        fwd_trail_segments = []
        
        # Add pulsing animation to the dot
        pulse = Succession(
            fwd_dot.animate.scale(1.5),
            fwd_dot.animate.scale(1/1.5),
        )
        self.play(FadeIn(fwd_dot), Write(z_labels[0]), pulse)
        self.next_slide()

        # Forward equation with fade and slide
        fwd_eq = MathTex(
            r"z_t = \sqrt{\bar{\alpha}_t}\,z_T + \sqrt{1-\bar{\alpha}_t}\,\varepsilon", color=BLUE
        ).scale(0.7).move_to(RIGHT * 2.5 + UP)
        self.play(
            FadeIn(fwd_eq, shift=UP)
        )
        self.play(
            fwd_dot.animate.set_color(YELLOW)
        )


        # Animate forward path with dashed segments and ripple effects
        for i, pos in enumerate(forward_positions[1:], 1):
            # Create new dashed segment with gradient
            segment = Line(forward_positions[i-1], pos, color=BLUE)
            dashed_segment = DashedVMobject(segment, num_dashes=15)
            fwd_trail_segments.append(dashed_segment)
            
            # Add ripple effect at each point
            ripple = Circle(radius=0.1, color=BLUE, fill_opacity=0.2)
            ripple.move_to(pos)
            
            self.play(
                fwd_dot.animate.move_to(pos),
                Create(dashed_segment),
                FadeIn(z_labels[i], shift=UP * 0.3),
                Create(ripple),
                ripple.animate.scale(3).fade(1),
                run_time=1.3
            )
        self.next_slide()

        # Noise cloud around z_1 with dynamic effect
        noise_cloud = Circle(radius=0.5, color=RED, fill_opacity=0.15).move_to(forward_positions[-1])
        noise_label = MathTex(r"z_1 \sim \mathcal{N}(0, I)", color=RED).scale(0.5).next_to(noise_cloud, DOWN)
        
        # Create noise particles with proper numpy reference
        particles = VGroup(*[
            Dot(
                point=noise_cloud.point_from_proportion(i/20),
                radius=0.02,
                color=RED
            ).shift(np.array([
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                0
            ]))
            for i in range(20)
        ])
        
        self.play(
            Create(noise_cloud),
            Write(noise_label),
            LaggedStart(*[
                FadeIn(p, shift=np.array([
                    np.random.normal(0, 0.1),
                    np.random.normal(0, 0.1),
                    0
                ]))
                for p in particles
            ], lag_ratio=0.05)
        )
        self.next_slide()

        # Reverse diffusion equation with transform effect
        rev_eq = MathTex(
            r"\hat{z}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \hat{z}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\hat{z}_t, t) \right)", color=GREEN
        ).scale(0.7).move_to(RIGHT * 3 + UP)
        self.play(
            Transform(fwd_eq, rev_eq),
            noise_cloud.animate.set_opacity(0.1),
            *[p.animate.fade(0.7) for p in particles]
        )
        self.next_slide()

        # Create noisy reverse positions with dynamic noise visualization
        noise_scale = 0.15
        reverse_positions = []
        noise_vectors = []  # Store noise vectors for visualization
        for pos in forward_positions[::-1]:
            noise = np.array([
                np.random.normal(0, noise_scale),
                np.random.normal(0, noise_scale),
                0
            ])
            noisy_pos = np.array(pos) + noise
            reverse_positions.append(noisy_pos)
            noise_vectors.append(noise)

        # Initialize reverse path with pulsing effect
        rev_dot = Dot(reverse_positions[0], color=GREEN)
        rev_trail_segments = []
        
        # Create reverse point labels with dynamic appearance
        z_hat_labels = []
        for i, pos in enumerate(reverse_positions):
            if i == 0:
                label = MathTex(r"\hat{z}_1", color=GREEN).scale(0.5)
            elif i == len(reverse_positions) - 1:
                label = MathTex(r"\hat{z}_T", color=GREEN).scale(0.5)
            else:
                label = MathTex(f"\\hat{{z}}_{i+1}", color=GREEN).scale(0.5)
            label.next_to(pos, DOWN)
            z_hat_labels.append(label)
        
        # Add initial pulse to reverse dot
        rev_pulse = Succession(
            rev_dot.animate.scale(1.5),
            rev_dot.animate.scale(1/1.5),
        )
        self.play(FadeIn(rev_dot), Write(z_hat_labels[0]), rev_pulse)

        # Animate reverse path with enhanced effects
        for i, pos in enumerate(reverse_positions[1:], 1):
            # Create new dashed segment
            segment = Line(reverse_positions[i-1], pos, color=GREEN)
            dashed_segment = DashedVMobject(segment, num_dashes=15)
            rev_trail_segments.append(dashed_segment)
            
            # Create noise visualization
            noise_arrow = Arrow(
                forward_positions[::-1][i],
                pos,
                buff=0.1,
                color=GREEN,
                stroke_opacity=0.3
            )
            
            # Add ripple effect
            ripple = Circle(radius=0.1, color=GREEN, fill_opacity=0.2)
            ripple.move_to(pos)
            
            self.play(
                rev_dot.animate.move_to(pos),
                Create(dashed_segment),
                FadeIn(z_hat_labels[i], shift=DOWN * 0.3),
                Create(ripple),
                ripple.animate.scale(3).fade(1),
                FadeIn(noise_arrow, rate_func=there_and_back),
                run_time=1.3
            )
        self.next_slide()

        # Final DDPM loss with dynamic appearance
        loss = MathTex(
            r"\mathcal{L}_{\mathrm{LDM}}(\theta) = \sum_{t=2}^{T_i} \mathbb{E}_{t,z_T,\varepsilon}\bigl\|\varepsilon - \varepsilon_\theta(z_t,t)\bigr\|^2"
        ).scale(0.7).move_to(RIGHT * 3 + DOWN * 2.5)
        
        # Highlight different parts of the loss
        loss_parts = [
            loss[0][i:j] for i, j in [(0, 14), (14, 20), (20, 37), (37, 45)]
        ]
        
        self.play(
            *[Write(part) for part in loss_parts],
            lag_ratio=0.5,
            run_time=2
        )
        self.next_slide()

#class KMeansMetricVisualization(Slide):
    def construct(self):
        # Title
        title = Text("K-means Clustering and Metric Calculation", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Create axes for latent space with grid
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            axis_config={
                "stroke_color": GRAY,
                "stroke_width": 1,
                "include_numbers": True,
                "numbers_to_exclude": [-3, -2, -1, 0, 1, 2, 3]
            },
            x_axis_config={"numbers_to_include": [-3, -2, -1, 0, 1, 2, 3]},
            y_axis_config={"numbers_to_include": [-3, -2, -1, 0, 1, 2, 3]},
        ).scale(0.8)
        axes.to_corner(LEFT + DOWN, buff=0.7)

        # Create grid manually
        x_range = np.arange(-4, 4.1, 1)
        y_range = np.arange(-4, 4.1, 1)
        
        grid = VGroup()
        # Vertical lines
        for x in x_range:
            line = Line(
                start=axes.c2p(x, -4),
                end=axes.c2p(x, 4),
                color=GRAY_D,
                stroke_opacity=0.3,
                stroke_width=0.5
            )
            grid.add(line)
        
        # Horizontal lines
        for y in y_range:
            line = Line(
                start=axes.c2p(-4, y),
                end=axes.c2p(4, y),
                color=GRAY_D,
                stroke_opacity=0.3,
                stroke_width=0.5
            )
            grid.add(line)

        self.play(Create(axes), Create(grid))
        self.wait(1)

        # Create prior distribution visualization - only 3σ circle
        prior_circle = Circle(radius=3 * 0.8, color=BLUE_E, fill_opacity=0.05)
        prior_circle.move_to(axes.get_center())
        prior_label = MathTex(r"p(z) = \mathcal{N}(0, I)", color=BLUE_E).scale(0.6)
        prior_label.next_to(prior_circle, UP, buff=0.1)
        
        self.play(Create(prior_circle), Write(prior_label))
        self.wait(1)

        # Explanation of sampling
        sampling_text = Text("Sampling points from N(0,1)", font_size=24, color=WHITE)
        sampling_text.move_to(RIGHT * 3 + UP * 2.5)
        self.play(Write(sampling_text))
        self.wait(1)

        # Generate random points from N(0,1)
        np.random.seed(42)  # For reproducibility
        n_points = 100
        points = np.random.normal(0, 1, (n_points, 2))
        
        # Create dots for points with coordinate display for some points
        dots = VGroup()
        coord_labels = VGroup()
        for i, (x, y) in enumerate(points):
            dot = Dot(axes.c2p(x, y), color=WHITE, radius=0.06)  # Slightly bigger dots
            dots.add(dot)
            # Add coordinates for some random points
            if i % 20 == 0:  # Show coordinates for every 20th point
                coord = MathTex(f"({x:.1f}, {y:.1f})", color=WHITE).scale(0.3)
                coord.next_to(dot, UR, buff=0.1)
                coord_labels.add(coord)

        self.play(
            LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.02),
            LaggedStart(*[FadeIn(label) for label in coord_labels], lag_ratio=0.02)
        )
        self.wait(1)

        # K-means clustering explanation
        kmeans_text = Text("K-means Clustering Process", font_size=24, color=WHITE)
        kmeans_text.move_to(RIGHT * 3 + UP * 2.5)
        self.play(
            Transform(sampling_text, kmeans_text),
            FadeOut(coord_labels)  # Remove coordinate labels before clustering
        )
        self.wait(1)

        # K-means clustering animation
        K = 3
        centroids = points[np.random.choice(n_points, K, replace=False)]
        centroid_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=MAROON_E, radius=0.12)  # Bigger centroid dots
            for x, y in centroids
        ])
        
        # Add centroid labels with coordinates
        centroid_labels = VGroup()
        centroid_coords = VGroup()
        for i, (x, y) in enumerate(centroids):
            label = MathTex(f"c_{i+1}", color=MAROON_E).scale(0.5)
            coord = MathTex(f"({x:.1f}, {y:.1f})", color=MAROON_E).scale(0.4)
            label.next_to(centroid_dots[i], UP, buff=0.1)
            coord.next_to(label, UP, buff=0.1)
            centroid_labels.add(label)
            centroid_coords.add(coord)
        
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in centroid_dots], lag_ratio=0.1),
            LaggedStart(*[Write(label) for label in centroid_labels], lag_ratio=0.1),
            LaggedStart(*[Write(coord) for coord in centroid_coords], lag_ratio=0.1)
        )
        self.wait(1)

        # Animate k-means iterations with enhanced visualization
        cluster_colors = [MAROON_E, TEAL_E, PURPLE_E]  # More distinct colors
        for iteration in range(3):
            # Assign points to clusters
            distances = np.array([[np.linalg.norm(p - c) for c in centroids] for p in points])
            clusters = np.argmin(distances, axis=1)
            
            # Create cluster lines with gradient and varying opacity based on distance
            cluster_lines = VGroup()
            for i, (x, y) in enumerate(points):
                c_x, c_y = centroids[clusters[i]]
                dist = np.linalg.norm([x - c_x, y - c_y])
                opacity = max(0.1, min(0.5, 1.0 - dist/4))  # Opacity decreases with distance
                line = Line(
                    axes.c2p(x, y),
                    axes.c2p(c_x, c_y),
                    color=cluster_colors[clusters[i]],
                    stroke_opacity=opacity
                )
                cluster_lines.add(line)
            
            # Show iteration number
            iter_text = Text(f"Iteration {iteration + 1}", font_size=24, color=WHITE)
            iter_text.move_to(RIGHT * 3 + UP * 2)
            self.play(
                Write(iter_text),
                Create(cluster_lines),
                run_time=1
            )
            self.wait(1)
            
            # Update centroids
            new_centroids = np.array([
                points[clusters == k].mean(axis=0) for k in range(K)
            ])
            
            # Update centroid positions and labels
            for i, (old, new) in enumerate(zip(centroids, new_centroids)):
                new_coord = MathTex(f"({new[0]:.1f}, {new[1]:.1f})", color=cluster_colors[i]).scale(0.4)
                new_coord.next_to(centroid_labels[i], UP, buff=0.1)
                self.play(
                    centroid_dots[i].animate.move_to(axes.c2p(new[0], new[1])),
                    centroid_labels[i].animate.next_to(centroid_dots[i], UP, buff=0.1),
                    Transform(centroid_coords[i], new_coord),
                    run_time=1
                )
            centroids = new_centroids
            
            self.play(FadeOut(cluster_lines), FadeOut(iter_text))
            self.wait(1)

        # Show final clusters with different colors and labels
        for k in range(K):
            cluster_points = points[clusters == k]
            for i, (x, y) in enumerate(cluster_points):
                self.play(
                    dots[np.where(clusters == k)[0][i]].animate.set_color(cluster_colors[k]).set_opacity(0.8),
                    run_time=0.05
                )

        # Add cluster labels
        cluster_labels = VGroup()
        for k in range(K):
            label = MathTex(f"C_{k+1}", color=cluster_colors[k]).scale(0.6)
            label.next_to(centroid_dots[k], DOWN, buff=0.1)
            cluster_labels.add(label)
        
        self.play(LaggedStart(*[Write(label) for label in cluster_labels], lag_ratio=0.2))
        self.wait(1)

        # Calculate and show precision matrices Mk with explanation
        precision_text = Text("Calculating Precision Matrices", font_size=24, color=WHITE)
        precision_text.move_to(RIGHT * 3 + UP * 2.5)
        self.play(Transform(kmeans_text, precision_text))
        self.wait(1)

        precision_matrices = []
        for k in range(K):
            cluster_points = points[clusters == k]
            # Simplified precision matrix calculation
            Mk = np.cov(cluster_points.T)
            precision_matrices.append(Mk)
            
            # Visualize Mk as an ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(Mk)
            angle = np.arctan2(eigenvectors[1,0], eigenvectors[0,0])
            ellipse = Ellipse(
                width=eigenvalues[0] * 2,
                height=eigenvalues[1] * 2,
                color=cluster_colors[k],
                fill_opacity=0.2
            )
            ellipse.move_to(axes.c2p(centroids[k][0], centroids[k][1]))
            ellipse.rotate(angle)
            
            # Add precision matrix label
            mk_label = MathTex(f"M_{k+1}", color=cluster_colors[k]).scale(0.5)
            mk_label.next_to(ellipse, RIGHT, buff=0.1)
            
            self.play(
                Create(ellipse),
                Write(mk_label),
                run_time=0.5
            )
        self.wait(1)

        # Select a random point and show metric calculation
        metric_text = Text("Calculating Local Metric", font_size=24, color=WHITE)
        metric_text.move_to(RIGHT * 3 + UP * 2.5)
        self.play(Transform(precision_text, metric_text))
        self.wait(1)

        # Set specific position for z instead of random
        random_point = np.array([-1.0, -3])  # Fixed position at (-1, -1)
        random_dot = Dot(axes.c2p(random_point[0], random_point[1]), color=GOLD_E, radius=0.08)
        random_label = MathTex("z", color=GOLD_E).scale(0.5).next_to(random_dot, UP, buff=0.1)
        self.play(FadeIn(random_dot), Write(random_label))
        self.wait(1)

        # Calculate weights and show metric
        distances_to_centroids = np.array([np.linalg.norm(random_point - c) for c in centroids])
        weights = np.exp(-distances_to_centroids / 2) / np.sum(np.exp(-distances_to_centroids / 2))
        
        # Show weight calculation with explanation
        weight_text = Text("Calculating Weights", font_size=24, color=WHITE)
        weight_text.move_to(RIGHT * 3 + UP * 2)
        self.play(Write(weight_text))
        
        weight_eq = MathTex(
            r"w_k(z) = \frac{e^{-\frac{\|z - c_k\|^2}{2}}}{\sum_{\ell=1}^K e^{-\frac{\|z - c_\ell\|^2}{2}}}",
            color=GOLD_E
        ).scale(0.7).move_to(RIGHT * 3 + UP * 1)
        self.play(Write(weight_eq))
        self.wait(1)

        # Show connections between z and centroids with weights
        connection_lines = VGroup()
        for k in range(K):
            line = DashedLine(
                random_dot.get_center(),
                centroid_dots[k].get_center(),
                color=cluster_colors[k],
                stroke_opacity=0.5
            )
            connection_lines.add(line)
            
            # Show weight value
            weight_value = MathTex(
                f"w_{k+1} = {weights[k]:.2f}",
                color=cluster_colors[k]
            ).scale(0.5)
            weight_value.next_to(line.get_center(), UP, buff=0.1)
            
            self.play(
                Create(line),
                Write(weight_value),
                run_time=0.5
            )
        self.wait(1)

        # Show final metric calculation with highlighting
        metric_eq = MathTex(
            r"G^{-1}(z) = \sum_{k=1}^K w_k(z) M_k",
            color=GOLD_E
        ).scale(0.7).move_to(RIGHT * 3 + DOWN * 0.5)
        
        # Create boxes for highlighting
        weight_terms = VGroup(*[
            MathTex(f"w_{k+1}", color=cluster_colors[k]).scale(0.7)
            for k in range(K)
        ]).arrange(RIGHT, buff=0.5).next_to(metric_eq, DOWN, buff=0.5)
        
        matrix_terms = VGroup(*[
            MathTex(f"M_{k+1}", color=cluster_colors[k]).scale(0.7)
            for k in range(K)
        ]).arrange(RIGHT, buff=0.5).next_to(weight_terms, DOWN, buff=0.5)
        
        self.play(Write(metric_eq))
        self.play(Write(weight_terms), Write(matrix_terms))
        self.wait(1)

        # Highlight each term in the metric calculation
        for k in range(K):
            # Highlight the connection
            self.play(
                connection_lines[k].animate.set_stroke(opacity=1),
                run_time=0.5
            )
            
            # Highlight the weight and matrix terms
            weight_box = SurroundingRectangle(weight_terms[k], color=cluster_colors[k], buff=0.1)
            matrix_box = SurroundingRectangle(matrix_terms[k], color=cluster_colors[k], buff=0.1)
            
            self.play(
                Create(weight_box),
                Create(matrix_box),
                run_time=0.5
            )
            
            # Fade back
            self.play(
                connection_lines[k].animate.set_stroke(opacity=0.5),
                FadeOut(weight_box),
                FadeOut(matrix_box),
                run_time=0.5
            )
        self.next_slide()

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(prior_circle),
            FadeOut(prior_label),
            FadeOut(dots),
            FadeOut(centroid_dots),
            FadeOut(centroid_labels),
            FadeOut(cluster_labels),
            FadeOut(random_dot),
            FadeOut(random_label),
            FadeOut(weight_text),
            FadeOut(weight_eq),
            FadeOut(metric_eq),
            FadeOut(weight_terms),
            FadeOut(matrix_terms),
            FadeOut(metric_text),
            FadeOut(connection_lines)
        )
        self.next_slide()

#class GUGUSReconstructionPipeline(MovingCameraScene, Slide):
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
        self.current_mobjects = VGroup()
        self.num_epochs = 5
        self.num_points = 5
        self.patient_data = self.generate_patient_data()
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
        self.next_slide()
    
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
        self.next_slide()
        self.play(Create(encoder), run_time=1.5)
        self.play(Write(encoder_law))
        self.play(
            Write(input_label),
            Write(output_label),
            Create(outline),
            run_time=1.5
        )
        self.next_slide()
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
        self.next_slide()
    
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
        self.next_slide()
        self.play(Write(equations))
        self.play(Create(dots[2]), Write(labels[2]))  # Start with middle point
        self.next_slide()
        self.play(future_eq.animate.set_color(future_color).scale(1.1))
        for i in range(3, len(dots)):
            self.play(
                Create(dots[i]),
                Write(labels[i]),
                Create(future_arrows[i-3]),  # Adjusted index for future arrows
                run_time=0.8
            )
        self.next_slide()
        self.play(future_eq.animate.scale(1/1.1))
        self.play(past_eq.animate.set_color(past_color).scale(1.1))
        for i, idx in enumerate(range(1, -1, -1)):
            self.play(
                Create(dots[idx]),
                Write(labels[idx]),
                Create(past_arrows[i]),  # Use i for past arrows index
                run_time=0.8
            )
        self.next_slide()
        self.play(past_eq.animate.scale(1/1.1))
        self.play(Write(legend))
        # Store new elements
        new_mobjects = VGroup(
            equations, dots, labels, future_arrows, past_arrows, legend, trajectory_metrics
        )
        self.current_mobjects = VGroup(*self.current_mobjects, *new_mobjects)
        self.remove(title)
        self.next_slide()
    
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
        self.next_slide()
        self.play(
            Create(decoder),
            Write(input_label),
            Write(output_label)
        )
        self.play(Write(decoder_law))
        self.play(Create(reconstructed_manifold))
        for i in range(len(reconstructed_points)):
            self.play(
                Create(reconstructed_points[i]),
                Write(reconstructed_labels[i]),
                run_time=0.5
            )
            if i < len(reconstructed_arrows):
                self.play(Create(reconstructed_arrows[i]), run_time=0.3)
        self.next_slide()
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
        self.next_slide()
        for i in range(len(original_points)):
            self.play(
                Create(original_points[i]),
                Write(original_labels[i]),
                run_time=0.5
            )
            if i < len(original_arrows):
                self.play(Create(original_arrows[i]), run_time=0.3)
        self.next_slide()
        for i in range(len(reconstructed_points)):
            self.play(
                Create(reconstructed_points[i]),
                Write(reconstructed_labels[i]),
                run_time=0.5
            )
            if i < len(reconstructed_arrows):
                self.play(Create(reconstructed_arrows[i]), run_time=0.3)
        self.next_slide()
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
        self.next_slide()
        self.play(Write(legend))
        # Store final state
        self.current_mobjects = VGroup(
            original_manifold, reconstructed_manifold,
            original_title, reconstructed_title,
            original_points, original_labels, original_arrows,
            reconstructed_points, reconstructed_labels, reconstructed_arrows,
            error_arrow, error_label, legend
        ) 
# PLACEMENT: Main scene class that orchestrates all components


#class GUGUSTrainingPipeline(MovingCameraScene, Slide):
    """Main scene for visualizing GUGUS model training dynamics with URIEM emphasis."""

    
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
        self.next_slide()
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
        encoder = TEncoderNetwork(input_dim=2, hidden_dims=[4, 8, 4])
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
        latent_space = TLatentSpace(width=8.0, height=8.0)
        latent_space.move_to(encoder.get_center() + RIGHT * 7)
        # Create diffusion process inside latent space
        diffusion = TDiffusionProcess(
            start_point=latent_space.get_center(),
            num_steps=5
        )
        # Position diffusion in bottom part of latent space
        diffusion.move_to(latent_space.get_center() + DOWN * 0.1)
        # Create initial reconstructed trajectory
        reconstructed_trajectory = self.create_trajectory(data)
        reconstructed_trajectory.move_to(manifold.get_center())
        # Create decoder network
        decoder = TDecoderNetwork(input_dim=2, hidden_dims=[4, 8, 4])
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
        loss_viz = TLossVisualization(max_width=4.0)
        loss_viz.move_to(diffusion.get_center() + DOWN * 6)
        # Show initial state
        self.play(
            Write(title),
            Create(manifold),
            Write(trajectory_labels),
            run_time=1
        )
        self.next_slide()
        # Show trajectory
        self.play(
            Create(trajectory),
            run_time=1.5
        )
        self.next_slide()
        # Show encoder with URIEM emphasis
        self.play(
            Create(encoder),
            Write(encoder_label),
            run_time=1.5
        )
        self.next_slide()
        # Show latent space with URIEM emphasis
        self.play(
            Create(latent_space),
            #Create(diffusion),
            run_time=1.5
        )
        self.next_slide()
        # Show decoder and reconstruction
        self.play(
            Create(decoder),
            Write(decoder_label),
            Create(reconstructed_manifold),
            Write(recon_label),
            run_time=1.5
        )
        self.next_slide()
        # Show loss visualization
        self.play(
            Create(loss_viz),
            run_time=1.5
        )
        self.next_slide()
        # Training animation
        first_diffusion = None  # Keep track of the first diffusion
        for epoch in range(self.num_epochs):
            progress = epoch / (self.num_epochs - 1)
            # Update latent space
            new_latent_space = TLatentSpace(
                width=8.0,
                height=8.0,
                epoch=epoch,
                max_epochs=self.num_epochs
            )
            new_latent_space.move_to(latent_space.get_center())
            # Update diffusion process
            new_diffusion = TDiffusionProcess(
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
        self.next_slide()
        # Final state
        final_text = Text("Training Complete on URIEM", color=COLORS["highlight"]).scale(0.7)
        final_text.next_to(title, DOWN, buff=0.4)
        self.play(Write(final_text))
        self.next_slide()
    
    def construct(self):
        """Main construction sequence."""
        self.show_training_step()
        self.wait(2) 

    def construct(self):
        # SLIDE 1: Title with dynamic entrance
        title = Text("Vanilla VAE — Training Phase", font_size=36)
        subtitle = Text("Understanding the Architecture", font_size=24, color=BLUE)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP, run_time=1)
        )
        self.next_slide()

        # SLIDE 2: Full Pipeline with dynamic build-up
        # Initialize components with initial opacity 0
        x = MathTex(r"x_j^{(i)}", color=YELLOW).scale(1.2).move_to(LEFT * 6.5)
        encoder = Rectangle(width=3, height=1.5, color=BLUE).move_to(LEFT * 4)
        enc_lbl = Text("Encoder", font_size=24, color=BLUE).next_to(encoder, UP, buff=0.1)
        enc1 = MathTex(r"q_{\phi}(z|x)", color=BLUE_C).scale(0.6).move_to(encoder.get_center() + UP * 0.3)
        enc2 = MathTex(r"\mathcal N(\mu_{\phi}(x_j^{(i)}), \sigma_{\phi}(x_j^{(i)}))", color=BLUE_C).scale(0.5).next_to(enc1, DOWN, buff=0.2)
        
        # Create latent space with gradient fill
        latent = RoundedRectangle(corner_radius=0.2, width=3, height=3, color=ORANGE)
        latent_fill = latent.copy().set_fill(color=ORANGE, opacity=0.1)
        latent_group = VGroup(latent, latent_fill).move_to(ORIGIN)
        lat_lbl = Text("Latent Space", font_size=24, color=ORANGE).next_to(latent, UP, buff=0.1)
        
        decoder = Rectangle(width=3, height=1.5, color=GREEN).move_to(RIGHT * 4)
        dec_lbl = Text("Decoder", font_size=24, color=GREEN).next_to(decoder, UP, buff=0.1)
        dec1 = MathTex(r"p_{\theta}(z|x)", color=GREEN_C).scale(0.6).move_to(decoder.get_center() + UP * 0.3)
        dec2 = MathTex(r"\mathcal N(\mu_{\theta}(z_j^{(i)}), diag(\sigma_{\theta}^2(z_j^{(i)})))", color=GREEN_C).scale(0.45).next_to(dec1, DOWN, buff=0.2)
        xhat = MathTex(r"\hat x_j^{(i)}", color=YELLOW).scale(1.2).move_to(RIGHT * 6.5)

        # Create arrows with gradient
        arr1 = Arrow(x.get_right(), encoder.get_left(), buff=0.1, color=BLUE)
        arr2 = Arrow(encoder.get_right(), latent.get_left(), buff=0.1, color=ORANGE)
        arr3 = Arrow(latent.get_right(), decoder.get_left(), buff=0.1, color=GREEN)
        arr4 = Arrow(decoder.get_right(), xhat.get_left(), buff=0.1, color=YELLOW)

        # Prior information with dynamic appearance
        prior_lbl = Text("Prior:", font_size=24).move_to(latent.get_center())
        zij = MathTex(r"z_j^{(i)}", color=ORANGE).scale(0.9).next_to(prior_lbl, UP, buff=0.2)
        prior = MathTex(r"p(z)=\mathcal N(0,I)", color=ORANGE).scale(0.8).next_to(prior_lbl, DOWN, buff=0.2)

        # Animate components sequentially with effects
        self.play(FadeOut(title_group, shift=UP))
        
        # Input data animation
        self.play(
            Write(x),
            Create(VGroup(encoder, enc_lbl)),
            run_time=0.5
        )
        
        # Encoder equations with typewriter effect
        self.play(
            AddTextLetterByLetter(enc1),
            AddTextLetterByLetter(enc2),
            run_time=0.5
        )
        
        # Arrow animation with trace effect
        self.play(
            ShowPassingFlash(arr1.copy().set_color(WHITE), time_width=0.5),
            Create(arr1)
        )

        # Latent space appearance with growing effect
        self.play(
            Create(latent),
            FadeIn(latent_fill),
            Write(lat_lbl),
            run_time=0.5
        )
        
        # Prior information with fade effect
        self.play(
            Write(prior_lbl),
            FadeIn(zij, shift=DOWN),
            FadeIn(prior, shift=UP),
            run_time=0.5
        )

        # Second arrow with pulse
        self.play(
            ShowPassingFlash(arr2.copy().set_color(WHITE), time_width=0.5),
            Create(arr2)
        )

        # Decoder appearance with build-up
        self.play(
            Create(VGroup(decoder, dec_lbl)),
            run_time=0.5
        )
        
        # Decoder equations with typewriter effect
        self.play(
            AddTextLetterByLetter(dec1),
            AddTextLetterByLetter(dec2),
            run_time=0.5
        )

        # Final arrows and output
        self.play(
            ShowPassingFlash(arr3.copy().set_color(WHITE), time_width=0.5),
            Create(arr3)
        )
        self.play(
            ShowPassingFlash(arr4.copy().set_color(WHITE), time_width=0.5),
            Create(arr4),
            Write(xhat)
        )

        self.next_slide()

        # SLIDE 3: Highlight Latent Jacobian with dynamic effects
        latgroup = VGroup(latent, lat_lbl)
        extraction_box1 = SurroundingRectangle(latgroup, color=RED, buff=0.3)
        jac_arrow1 = Arrow(extraction_box1.get_bottom(), extraction_box1.get_bottom() + DOWN * 1.0, color=RED)
        jac_label1 = MathTex(r"c_k \;=\;\frac{1}{|\mathcal{I}k|}\sum_{(i,j)\in\mathcal{I}_k} z_j^{(i)}", color=RED).scale(0.7)
        
        # Add glow effect to the box
        glow = extraction_box1.copy().set_stroke(color=RED, opacity=0.5, width=10)
        
        # Animate highlight with glow
        self.play(
            Create(extraction_box1),
            FadeIn(glow, rate_func=there_and_back),
            run_time=0.5
        )
        
        # Animate arrow and label with dynamic effect
        self.play(
            Create(jac_arrow1),
            Write(jac_label1.next_to(jac_arrow1, DOWN, buff=0.1)),
            run_time=0.5
        )

        centro = Text("Centroids via k-means:", font_size=24, color=RED)
        self.play(
            Write(centro.next_to(extraction_box1, UP, buff=0.1)),
            jac_label1.animate.set_color(RED),
            run_time=0.5
        )

        # Decoder highlight with similar effects
        decgroup = VGroup(decoder, dec_lbl)
        extraction_box2 = SurroundingRectangle(decgroup, color=YELLOW, buff=0.3)
        glow2 = extraction_box2.copy().set_stroke(color=YELLOW, opacity=0.5, width=10)
        
        self.play(
            Create(extraction_box2),
            FadeIn(glow2, rate_func=there_and_back),
            run_time=0.5
        )

        jac_arrow2 = Arrow(extraction_box2.get_bottom(), extraction_box2.get_bottom() + DOWN * 1.0, color=YELLOW)
        jac_label2 = MathTex(r"J_\theta(z)", color=YELLOW).scale(0.7)
        
        self.play(
            Create(jac_arrow2),
            Write(jac_label2.next_to(jac_arrow2, DOWN, buff=0.1)),
            run_time=0.5
        )

        jaco = Text("Access to Jacobians:", font_size=24, color=YELLOW)
        self.play(
            Write(jaco.next_to(extraction_box2, UP, buff=0.1)),
            run_time=0.5
        )

        # Final equation with dynamic build-up
        eqm = MathTex(r"M_k =J_\theta(c_k)^{\!T}\,J_\theta(c_k)", color=YELLOW).scale(0.7)
        self.play(
            Write(eqm.next_to(jac_label2, DOWN, buff=0.1)),
            run_time=0.5
        )

        self.next_slide()

        # SLIDE 5: ELBO
        #elbo = MathTex(
        #    r"\mathcal L(x)=\mathbb E_{q(z|x)}[\log p(x|z)]"
        #    r"-\mathrm{KL}(q\|p)"
        #).scale(0.8).to_edge(DOWN)
        #self.remove(prior, kl)
        #self.play(FadeIn(elbo)) 

        # SLIDE 6: Metric Extraction
         # Clean up VAE components with fade out effect
        vae_components = VGroup(
            x, encoder, enc_lbl, enc1, enc2,
            latent, latent_fill, lat_lbl, zij,
            decoder, dec_lbl, dec1, dec2,
            xhat, arr1, arr2, arr3, arr4,
            prior_lbl, prior,
            extraction_box1, jac_arrow1, jac_label1,
            extraction_box2, jac_arrow2, jac_label2,
            centro, jaco, eqm,
            glow, glow2
        )
        
        self.play(
            *[FadeOut(obj, shift=DOWN * 0.5) for obj in vae_components],
            run_time=0.5
        )
        
        met_title = Text("Metric Extraction", font_size=36).to_edge(UP)
        self.play(Write(met_title), run_time=1)

        # Centroid equation with dynamic build-up
        jac_label1 = MathTex(r"c_k \;=\;\frac{1}{|\mathcal{I}k|}\sum_{(i,j)\in\mathcal{I}_k} z_j^{(i)}", color=RED).scale(0.7)
        jac_label1.move_to(UP * 2 + LEFT * 5)
        self.play(Write(jac_label1), run_time=0.5)

        # Precision matrix equation with highlight
        eqm = MathTex(r"M_k =J_\theta(c_k)^{\!T}\,J_\theta(c_k)", color=YELLOW).scale(0.7)
        eqm.next_to(jac_label1, DOWN, buff=0.5)
        self.play(Write(eqm), run_time=0.5)

        # Add inverse covariance with flash effect
        eqm2 = MathTex(r" =\Sigma_k^{-1}\; }", color=YELLOW).scale(0.7)
        eqm2.next_to(eqm, RIGHT, buff=0.1)
        self.play(
            Write(eqm2),
            Flash(eqm2, color=YELLOW, line_length=0.2),
            run_time=0.5
        )
        eqmgroup = VGroup(eqm, eqm2)

        # Weight equation with sequential animation
        wk = MathTex(
            r"w_k(z_{j}^{(i)})\;=\;\frac{e^{-\frac{\|z_{j}^{(i)} - c_k\|^2}{2\,\lambda\,T}}}"
            r"{\displaystyle \sum_{\ell=1}^{K} e^{-\frac{\|z_{j}^{(i)} - c_{\ell}\|^2}{2\,\lambda\,T}}}",
            color=ORANGE
        ).scale(0.7)
        wk.next_to(eqm, DOWN, buff=0.5)
        self.play(Write(wk), run_time=1)

        # Arrange equations
        all_group = VGroup(jac_label1, eqmgroup, wk).arrange(
            DOWN, 
            aligned_edge=LEFT, 
            buff=0.5
        ).move_to(LEFT * 4 + UP * 0.5)

        # Add explanations with dynamic arrows
        centroids = Text("Centroids obtained via k-means", font_size=24, color=WHITE)
        centroids.next_to(jac_label1, RIGHT * 5, buff=0.5)
        arr6 = Arrow(jac_label1.get_right(), centroids.get_left(), buff=0.2)
        self.play(
            Create(arr6),
            Write(centroids),
            run_time=0.5
        )

        mk2 = Paragraph(
            "The precision matrix (the inverse covariance) of \nlocal latent fluctuations",
            font_size=24,
            color=WHITE,
            line_spacing=0.3,
            width=6
        ).next_to(eqmgroup, RIGHT * 4, buff=0.5)
        arr7 = Arrow(eqmgroup.get_right(), mk2.get_left(), buff=0.2)
        self.play(
            Create(arr7),
            AddTextLetterByLetter(mk2),
            run_time=0.5
        )

        wk2 = Paragraph(
            "The weights of the local latent fluctuations",
            font_size=24,
            color=WHITE,
            line_spacing=0.3,
            width=6
        ).next_to(wk, RIGHT * 4, buff=0.5)
        arr8 = Arrow(wk.get_right(), wk2.get_left(), buff=0.2)
        self.play(
            Create(arr8),
            AddTextLetterByLetter(wk2),
            run_time=0.5
        )

        # Parameters with highlighting
        lambda_def = MathTex(
            r"\lambda > 0:",
            r"\text{ kernel bandwidth}",
            tex_to_color_map={r"\lambda": ORANGE, "bandwidth": ORANGE}
        ).scale(0.5)
        lambda_def.next_to(wk, DOWN, buff=0.4).align_to(wk, LEFT)
        self.play(
            Write(lambda_def),
            Flash(lambda_def[0], color=ORANGE, line_length=0.2),
            run_time=0.5
        )

        T_def = MathTex(
            r"T > 0:",
            r"\text{ softmax temperature}",
            tex_to_color_map={r"T": ORANGE, "temperature": ORANGE}
        ).scale(0.5)
        T_def.next_to(lambda_def, DOWN, buff=0.2).align_to(lambda_def, LEFT)
        self.play(
            Write(T_def),
            Flash(T_def[0], color=ORANGE, line_length=0.2),
            run_time=0.5
        )

        # Final equation with box animation
        eq = MathTex(
            r"\boxed{"
            r"G^{-1}(z_{j}^{(i)}) \;=\; \sum_{k=1}^K w_k(z_{j}^{(i)}) \, M_k \;\Longrightarrow\;G(z_{j}^{(i)})\approx\bigl(G^{-1}(z_{j}^{(i)})+\varepsilon I\bigr)^{-1}\!"
            r"}",
            tex_to_color_map={
                r"G^{-1}(z_{i}^j)": WHITE,
                r"G(z_{i}^j)": WHITE,
                r"w_k(z_{i}^j)": WHITE,
                r"M_k": WHITE,
                r"\Longrightarrow": WHITE,
                r"\sum": WHITE,
            }
        ).scale(0.6).move_to(RIGHT * 2.4 + DOWN * 2.6)
        
        box = SurroundingRectangle(eq, buff=0.2)
        self.play(Write(eq), run_time=1)
        self.play(Create(box), run_time=0.5)
        self.play(FadeOut(box), run_time=0.5)

        self.next_slide()
        self.wait(2)

        # Clean up everything
        final_components = VGroup(
            met_title, all_group,
            centroids, arr6, mk2, arr7, wk2, arr8,
            lambda_def, T_def, eq
        )
        
        self.play(
            *[FadeOut(obj, shift=DOWN * 0.5) for obj in final_components],
            run_time=0.5
        )

        self.next_slide()

        # Start VAE Visualization section
        title = Text("Vanilla VAE: Encoding and Decoding", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Create three sections for input, latent, and output spaces
        input_frame = Rectangle(height=4, width=3, color=YELLOW)
        input_frame.move_to(LEFT * 4)
        latent_frame = Rectangle(height=4, width=3, color=RED)
        latent_frame.move_to(ORIGIN)
        output_frame = Rectangle(height=4, width=3, color=ORANGE)
        output_frame.move_to(RIGHT * 4)

        # Labels for each space
        input_label = MathTex(r"x_j \sim \text{Data}", color=YELLOW).next_to(input_frame, UP)
        latent_label = MathTex(r"z_j \sim q_\phi(z|x)", color=RED).next_to(latent_frame, UP)
        output_label = MathTex(r"\hat{x}_j \sim p_\theta(x|z)", color=ORANGE).next_to(output_frame, UP)

        self.play(
            Create(input_frame),
            Create(latent_frame),
            Create(output_frame),
            Write(input_label),
            Write(latent_label),
            Write(output_label)
        )
        self.next_slide()

        # Create manifold-like input figure (swiss roll-like shape)
        t = np.linspace(0, 4*np.pi, 100)
        input_points = []
        for theta in t:
            # Create a curved manifold shape
            r = 0.3 * (1 + 0.2 * np.cos(3*theta))
            x = r * np.cos(theta)
            y = r * np.sin(theta) + 0.1 * np.cos(5*theta)
            input_points.append([x, y, 0])

        # Add some perpendicular variation to create a 2D manifold
        manifold_points = []
        for p in input_points:
            # Add points perpendicular to the curve to create width
            for w in np.linspace(-0.15, 0.15, 5):
                # Calculate tangent vector
                dx = -p[1]
                dy = p[0]
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    dx, dy = dx/norm, dy/norm
                    manifold_points.append([
                        p[0] + w*dx,
                        p[1] + w*dy,
                        0
                    ])

        input_shape = VMobject(color=YELLOW, fill_opacity=0.2)
        input_shape.set_points_smoothly([
            input_frame.get_center() + np.array(p) for p in manifold_points[::5]
        ])
        
        # Add some points on the manifold
        input_dots = VGroup(*[
            Dot(input_frame.get_center() + np.array(p), color=YELLOW_E, radius=0.05)
            for p in manifold_points[::25]
        ])
        
        self.play(
            Create(input_shape),
            LaggedStart(*[FadeIn(dot) for dot in input_dots], lag_ratio=0.1)
        )

        # Create latent space visualization
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            axis_config={"stroke_color": GRAY, "stroke_width": 1},
        ).scale(0.7)
        axes.move_to(latent_frame.get_center())
        
        # Prior distribution
        prior_circle = Circle(radius=1.2, color=RED, fill_opacity=0.1)
        prior_circle.move_to(axes.get_center())
        prior_label = MathTex(r"p(z) = \mathcal{N}(0, I)", color=RED).scale(0.6)
        prior_label.next_to(prior_circle, UP, buff=0.1)

        self.play(
            Create(axes),
            Create(prior_circle),
            Write(prior_label)
        )

        # Encode points into latent space with noise clouds (constrained within prior)
        encoded_points = []
        for dot in input_dots:
            # Get relative position and scale to fit within prior
            pos = dot.get_center() - input_frame.get_center()
            # Normalize to ensure points stay within prior radius
            norm = np.sqrt(pos[0]**2 + pos[1]**2)
            if norm > 0:
                scale = min(1.0, prior_circle.radius / norm)
                encoded_pos = axes.c2p(pos[0]*scale, pos[1]*scale)
                encoded_points.append(encoded_pos)

        encoded_clouds = VGroup(*[
            Circle(radius=0.2, color=BLUE, fill_opacity=0.15).move_to(pos)
            for pos in encoded_points
        ])

        encoded_dots = VGroup(*[
            Dot(pos, color=BLUE)
            for pos in encoded_points
        ])

        # Add particles around encoded points (constrained within prior)
        for cloud in encoded_clouds:
            cloud_center = cloud.get_center()
            particles = VGroup(*[
                Dot(
                    point=cloud_center + np.array([
                        np.random.normal(0, 0.1),
                        np.random.normal(0, 0.1),
                        0
                    ]) * 0.8,  # Scale factor to keep particles closer to center
                    radius=0.02,
                    color=BLUE_A
                )
                for _ in range(10)
            ])
            # Ensure particles stay within prior
            for particle in particles:
                pos = particle.get_center() - axes.get_center()
                norm = np.sqrt(pos[0]**2 + pos[1]**2)
                if norm > prior_circle.radius:
                    scale = prior_circle.radius / norm
                    particle.move_to(axes.get_center() + pos * scale)
            cloud.particles = particles

        # Animate encoding
        encode_arrow = Arrow(input_frame.get_right(), latent_frame.get_left(), color=BLUE, buff=0.2)
        encode_text = MathTex(r"q_\phi(z|x)", color=BLUE).next_to(encode_arrow, UP, buff=0.1).scale(0.5)

        self.play(
            Create(encode_arrow),
            Write(encode_text)
        )

        for cloud, dot in zip(encoded_clouds, encoded_dots):
            self.play(
                Create(cloud),
                FadeIn(dot),
                LaggedStart(*[FadeIn(p) for p in cloud.particles], lag_ratio=0.05),
                run_time=0.5
            )

        # Create output manifold (slightly distorted version of input)
        output_points = []
        for p in manifold_points:
            noise = np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05), 0])
            output_points.append(np.array(p) + noise)

        output_shape = VMobject(color=ORANGE, fill_opacity=0.2)
        output_shape.set_points_smoothly([
            output_frame.get_center() + np.array(p) for p in output_points[::5]
        ])
        
        output_dots = VGroup(*[
            Dot(output_frame.get_center() + np.array(p), color=ORANGE, radius=0.05)
            for p in output_points[::25]
        ])

        # Animate decoding
        decode_arrow = Arrow(latent_frame.get_right(), output_frame.get_left(), color=GREEN, buff=0.2)
        decode_text = MathTex(r"p_\theta(x|z)", color=GREEN).next_to(decode_arrow, UP, buff=0.1).scale(0.5)

        self.play(
            Create(decode_arrow),
            Write(decode_text)
        )

        self.play(
            Create(output_shape),
            LaggedStart(*[FadeIn(dot) for dot in output_dots], lag_ratio=0.1)
        )

        # Add loss terms
        kl_div = MathTex(r"\text{KL}(q_\phi(z|x) \| p(z))", color=WHITE).scale(0.7)
        kl_div.next_to(latent_frame, DOWN, buff=0.2)
        
        rec_loss = MathTex(r"\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]", color=WHITE).scale(0.7)
        rec_loss.next_to(kl_div, DOWN, buff=0.2)

        self.play(Write(kl_div))
        self.play(Write(rec_loss))

        self.next_slide()
        self.wait(2)

        # Clean up
        vae_components = VGroup(
            title, input_frame, latent_frame, output_frame,
            input_label, latent_label, output_label,
            input_shape, input_dots,
            axes, prior_circle, prior_label,
            encoded_clouds, encoded_dots,
            encode_arrow, encode_text,
            output_shape, output_dots,
            decode_arrow, decode_text,
            kl_div, rec_loss
        )
        
        for cloud in encoded_clouds:
            vae_components.add(cloud.particles)

        self.play(
            *[FadeOut(obj, shift=DOWN * 0.5) for obj in vae_components],
            run_time=1.5
        )

        self.next_slide()

        # Start Diffusion section
        title = Text("Latent Diffusion: Pre-training", font_size=42).to_edge(UP)
        self.play(Write(title))
        self.next_slide()
        
        # Prompt line
        embed_prompt = MathTex(
            r"\text{1 - For each sequence }(i), \text{ encode only its final observation:}"
        ).scale(0.8).next_to(title, DOWN, buff=0.6)

        # Equation line, broken into two parts for readability
        embed_eq = MathTex(
            r"z_{T_i}^{(i)} \sim q_{\phi}\bigl(z \mid x_{T_i}^{(i)}\bigr)",
            r"= \mathcal{N}\bigl(\mu_{\phi}(x_{T_i}^{(i)}),\,\mathrm{diag}\bigl(\sigma_{\phi}^2(x_{T_i}^{(i)})\bigr)\bigr)",
            tex_to_color_map={
                r"z_{T_i}^{(i)}": YELLOW,
                r"q_{\phi}": BLUE,
                r"\mu_{\phi}": GREEN,
                r"\sigma_{\phi}": RED
            }
        ).scale(0.8).next_to(embed_prompt, DOWN, buff=0.4)
        diffusiongroup1 = VGroup(embed_prompt, embed_eq)
        embed_box = SurroundingRectangle(diffusiongroup1, color=RED, buff=0.3)

        self.play(Write(embed_prompt), Write(embed_eq), Write(embed_box))
        self.next_slide()

        # Forward noising (diffusion) prompt
        diffusion_prompt = MathTex(
            r"\text{2 - Forward diffusion (noising) at step } t:",
        ).scale(0.8).next_to(embed_eq, DOWN, buff=0.8)
        diffusion_eq = MathTex(
            r"z_t^{(i)} = \sqrt{\bar{\alpha}_t} z_{T_i}^{(i)} + \sqrt{1 - \bar{\alpha}_t} \varepsilon",
            r", \quad \varepsilon \sim \mathcal{N}(0,I)",
            tex_to_color_map={
                r"\sqrt{\bar{\alpha}_t}": BLUE,
                r"\sqrt{1 - \bar{\alpha}_t}": BLUE,
                r"\varepsilon": GREEN,
                r"\mathcal{N}(0,I)": GREEN
            }
        ).scale(0.7).next_to(diffusion_prompt, DOWN, buff=0.4)
        diffusiongroup2 = VGroup(diffusion_prompt, diffusion_eq)
        diffusion_box = SurroundingRectangle(diffusiongroup2, color=BLUE, buff=0.3)


        # Forward noising (diffusion) prompt
        diffusion_prompt2 = MathTex(
            r"\text{3 - Reverse diffusion (denoising) at step } t:",
        ).scale(0.8).next_to(diffusion_eq, DOWN, buff=0.8)
        diffusion_eq2 = MathTex(
            r"z_{t-1}^{(i)} = \frac{1}{\sqrt{\alpha_t}} \left(z_t^{(i)} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(z_t^{(i)}, t) \right) + \sigma_t^{(i)} \, u",
            r", \quad u \sim \mathcal{N}(0,I)",
            tex_to_color_map={
                r"\sqrt{\bar{\alpha}_t}": BLUE,
                r"\sqrt{1 - \bar{\alpha}_t}": BLUE,
                r"\varepsilon": GREEN,
                r"\mathcal{N}(0,I)": GREEN
            }
        ).scale(0.7).next_to(diffusion_prompt2, DOWN, buff=0.4)
        diffusiongroup3 = VGroup(diffusion_prompt2, diffusion_eq2)
        diffusion_box2 = SurroundingRectangle(diffusiongroup3, color=GREEN, buff=0.3)
        # Play everything in one go
        self.play(
            Write(diffusion_prompt),
            Write(diffusion_eq),
            run_time=2
        )
        self.play(Write(diffusion_box))
        self.next_slide()

        self.play(
            Write(diffusion_prompt2),
            Write(diffusion_eq2),
            run_time=2
        )
        self.play(Write(diffusion_box2))
        self.next_slide()

        self.remove(embed_prompt, embed_eq, diffusion_prompt, diffusion_eq, diffusion_prompt2, diffusion_eq2)
        self.remove(embed_box, diffusion_box, diffusion_box2)
        self.remove(title)

    

        title = Text("Real Latent Diffusion Process", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.next_slide()

        # Axes setup
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"stroke_color": GRAY, "stroke_width": 1},
        ).scale(0.9)
        axes.to_corner(LEFT + DOWN, buff=0.7)
        self.play(Create(axes))
        self.next_slide()

        # Time axis
        T = 4
        time_line = NumberLine(
            x_range=[1, T, 1],
            length=axes.width,
            include_numbers=True,
            tick_size=0.1,
        ).next_to(axes, DOWN, buff=1.2)
        t_label = Text("t", font_size=24).next_to(time_line, LEFT, buff=0.2)
        pointer = Dot(time_line.n2p(T), color=YELLOW)
        self.play(Create(time_line), Write(t_label), FadeIn(pointer))
        self.next_slide()

        # Forward diffusion (noising) z_T to z_1
        forward_positions = [
            axes.c2p(0.0, 0.0),
            axes.c2p(-0.3, 0.4),
            axes.c2p(-0.4, 0.0),
            axes.c2p(-0.7, 0.4),
            axes.c2p(-1.2, -0.7),
        ]

        forward_positions.reverse()

        # Create point labels with fade-in effect
        z_labels = []
        for i, pos in enumerate(forward_positions):
            if i == 0:
                label = MathTex("z_T", color=BLUE).scale(0.5)
            elif i == len(forward_positions) - 1:
                label = MathTex("z_1", color=BLUE).scale(0.5)
            else:
                label = MathTex(f"z_{len(forward_positions)-i}", color=BLUE).scale(0.5)
            label.next_to(pos, UP)
            z_labels.append(label)

        # Initialize forward trail segments with pulsing dot
        fwd_dot = Dot(forward_positions[0], color=BLUE)
        fwd_trail_segments = []
        
        # Add pulsing animation to the dot
        pulse = Succession(
            fwd_dot.animate.scale(1.5),
            fwd_dot.animate.scale(1/1.5),
        )
        self.play(FadeIn(fwd_dot), Write(z_labels[0]), pulse)
        self.next_slide()

        # Forward equation with fade and slide
        fwd_eq = MathTex(
            r"z_t = \sqrt{\bar{\alpha}_t}\,z_T + \sqrt{1-\bar{\alpha}_t}\,\varepsilon", color=BLUE
        ).scale(0.7).move_to(RIGHT * 2.5 + UP)
        self.play(
            FadeIn(fwd_eq, shift=UP)
        )
        self.play(
            fwd_dot.animate.set_color(YELLOW)
        )


        # Animate forward path with dashed segments and ripple effects
        for i, pos in enumerate(forward_positions[1:], 1):
            # Create new dashed segment with gradient
            segment = Line(forward_positions[i-1], pos, color=BLUE)
            dashed_segment = DashedVMobject(segment, num_dashes=15)
            fwd_trail_segments.append(dashed_segment)
            
            # Add ripple effect at each point
            ripple = Circle(radius=0.1, color=BLUE, fill_opacity=0.2)
            ripple.move_to(pos)
            
            self.play(
                fwd_dot.animate.move_to(pos),
                Create(dashed_segment),
                FadeIn(z_labels[i], shift=UP * 0.3),
                Create(ripple),
                ripple.animate.scale(3).fade(1),
                run_time=1.3
            )
        self.next_slide()

        # Noise cloud around z_1 with dynamic effect
        noise_cloud = Circle(radius=0.5, color=RED, fill_opacity=0.15).move_to(forward_positions[-1])
        noise_label = MathTex(r"z_1 \sim \mathcal{N}(0, I)", color=RED).scale(0.5).next_to(noise_cloud, DOWN)
        
        # Create noise particles with proper numpy reference
        particles = VGroup(*[
            Dot(
                point=noise_cloud.point_from_proportion(i/20),
                radius=0.02,
                color=RED
            ).shift(np.array([
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                0
            ]))
            for i in range(20)
        ])
        
        self.play(
            Create(noise_cloud),
            Write(noise_label),
            LaggedStart(*[
                FadeIn(p, shift=np.array([
                    np.random.normal(0, 0.1),
                    np.random.normal(0, 0.1),
                    0
                ]))
                for p in particles
            ], lag_ratio=0.05)
        )
        self.next_slide()

        # Reverse diffusion equation with transform effect
        rev_eq = MathTex(
            r"\hat{z}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \hat{z}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\hat{z}_t, t) \right)", color=GREEN
        ).scale(0.7).move_to(RIGHT * 3 + UP)
        self.play(
            Transform(fwd_eq, rev_eq),
            noise_cloud.animate.set_opacity(0.1),
            *[p.animate.fade(0.7) for p in particles]
        )
        self.next_slide()

        # Create noisy reverse positions with dynamic noise visualization
        noise_scale = 0.15
        reverse_positions = []
        noise_vectors = []  # Store noise vectors for visualization
        for pos in forward_positions[::-1]:
            noise = np.array([
                np.random.normal(0, noise_scale),
                np.random.normal(0, noise_scale),
                0
            ])
            noisy_pos = np.array(pos) + noise
            reverse_positions.append(noisy_pos)
            noise_vectors.append(noise)

        # Initialize reverse path with pulsing effect
        rev_dot = Dot(reverse_positions[0], color=GREEN)
        rev_trail_segments = []
        
        # Create reverse point labels with dynamic appearance
        z_hat_labels = []
        for i, pos in enumerate(reverse_positions):
            if i == 0:
                label = MathTex(r"\hat{z}_1", color=GREEN).scale(0.5)
            elif i == len(reverse_positions) - 1:
                label = MathTex(r"\hat{z}_T", color=GREEN).scale(0.5)
            else:
                label = MathTex(f"\\hat{{z}}_{i+1}", color=GREEN).scale(0.5)
            label.next_to(pos, DOWN)
            z_hat_labels.append(label)
        
        # Add initial pulse to reverse dot
        rev_pulse = Succession(
            rev_dot.animate.scale(1.5),
            rev_dot.animate.scale(1/1.5),
        )
        self.play(FadeIn(rev_dot), Write(z_hat_labels[0]), rev_pulse)

        # Animate reverse path with enhanced effects
        for i, pos in enumerate(reverse_positions[1:], 1):
            # Create new dashed segment
            segment = Line(reverse_positions[i-1], pos, color=GREEN)
            dashed_segment = DashedVMobject(segment, num_dashes=15)
            rev_trail_segments.append(dashed_segment)
            
            # Create noise visualization
            noise_arrow = Arrow(
                forward_positions[::-1][i],
                pos,
                buff=0.1,
                color=GREEN,
                stroke_opacity=0.3
            )
            
            # Add ripple effect
            ripple = Circle(radius=0.1, color=GREEN, fill_opacity=0.2)
            ripple.move_to(pos)
            
            self.play(
                rev_dot.animate.move_to(pos),
                Create(dashed_segment),
                FadeIn(z_hat_labels[i], shift=DOWN * 0.3),
                Create(ripple),
                ripple.animate.scale(3).fade(1),
                FadeIn(noise_arrow, rate_func=there_and_back),
                run_time=1.3
            )
        self.next_slide()

        # Final DDPM loss with dynamic appearance
        loss = MathTex(
            r"\mathcal{L}_{\mathrm{LDM}}(\theta) = \sum_{t=2}^{T_i} \mathbb{E}_{t,z_T,\varepsilon}\bigl\|\varepsilon - \varepsilon_\theta(z_t,t)\bigr\|^2"
        ).scale(0.7).move_to(RIGHT * 3 + DOWN * 2.5)
        
        # Highlight different parts of the loss
        loss_parts = [
            loss[0][i:j] for i, j in [(0, 14), (14, 20), (20, 37), (37, 45)]
        ]
        
        self.play(
            *[Write(part) for part in loss_parts],
            lag_ratio=0.5,
            run_time=2
        )
        self.next_slide()
        self.clear()
        
        self.clear()
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
        self.clear()
        self.show_training_step()
        self.wait(2) 


