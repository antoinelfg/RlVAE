"""
RlVAE Persistent Scheme Component
================================

Reusable persistent scheme that shows the RlVAE pipeline overview
and highlights the current section being discussed.
"""

from manim import *
from .color_scheme import COLOR_SCHEME

class PersistentRlVAEScheme(VGroup):
    """Persistent scheme showing RlVAE pipeline with flow-based progression."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_section = None
        self.create_scheme()
    
    def create_scheme(self):
        """Create the persistent scheme with flow-based progression."""
        # Main container
        self.scheme_box = Rectangle(
            width=6, height=4,
            fill_opacity=0.1,
            stroke_color=COLOR_SCHEME["text"],
            stroke_width=1
        )
        
        # Stage 1: Foundation (top row)
        self.stage1_blocks = {
            "data_input": self.create_block("Data Input\nTrajectories", LEFT * 4.5, UP * 1.2),
            "vanilla_vae": self.create_block("Vanilla VAE\nq_φ(z|x)", LEFT * 2.5, UP * 1.2),
            "metric_g0": self.create_block("Metric G₀\nG₀(z) Matrix", LEFT * 0.5, UP * 1.2)
        }
        
        # Stage 2: Riemannian Prior (middle-left)
        self.stage2_blocks = {
            "riemannian_prior": self.create_block("Riemannian Prior\np(z|G₀)", LEFT * 4.5, ORIGIN),
            "initial_z0": self.create_block("Initial z₀\nSample from Prior", LEFT * 2.5, ORIGIN),
            "flow_sequence": self.create_block("Flow Sequence\nNormalizing Flows", LEFT * 0.5, ORIGIN)
        }
        
        # Stage 3: Temporal Evolution (middle-right)
        self.stage3_blocks = {
            "z0_to_z1": self.create_block("z₀ → z₁\nFlow Step 1", RIGHT * 1.5, ORIGIN),
            "z1_to_z2": self.create_block("z₁ → z₂\nFlow Step 2", RIGHT * 3.5, ORIGIN),
            "zt_minus_1_to_zt": self.create_block("z_{T-1} → z_T\nFinal Step", RIGHT * 5.5, ORIGIN),
            "decoder": self.create_block("Decoder\np_θ(x|z_T)", RIGHT * 7.5, ORIGIN),
            "reconstruction": self.create_block("Reconstruction\nOutput Sequence", RIGHT * 9.5, ORIGIN)
        }
        
        # Stage 4: Evaluation (bottom row)
        self.stage4_blocks = {
            "geometric_loss": self.create_block("Geometric Loss\nFlow-based", LEFT * 2.5, DOWN * 1.2),
            "evaluation": self.create_block("Evaluation\nSequence Quality", LEFT * 0.5, DOWN * 1.2),
            "sequence_quality": self.create_block("Sequence Quality\nFinal Metrics", RIGHT * 1.5, DOWN * 1.2)
        }
        
        # Combine all blocks
        self.all_blocks = {
            **self.stage1_blocks,
            **self.stage2_blocks,
            **self.stage3_blocks,
            **self.stage4_blocks
        }
        
        # Create connecting arrows emphasizing flow progression
        self.arrows = self.create_flow_arrows()
        
        # Add stage labels
        self.stage_labels = self.create_stage_labels()
        
        # Add title
        self.title = Text("RlVAE: Flow-Based Sequence Progression", 
                         font_size=10, color=COLOR_SCHEME["text"])
        self.title.next_to(self.scheme_box, UP, buff=0.1)
        
        self.add(self.scheme_box, self.title, *self.all_blocks.values(), 
                *self.arrows, *self.stage_labels)
    
    def create_block(self, text, position, offset):
        """Create a block with text."""
        block = Rectangle(
            width=1.4, height=0.7,
            fill_opacity=0.2,
            stroke_color=COLOR_SCHEME["text"],
            stroke_width=1
        )
        block.move_to(position + offset)
        
        # Add text
        text_obj = Text(text, font_size=5, color=COLOR_SCHEME["text"])
        text_obj.move_to(block.get_center())
        
        return VGroup(block, text_obj)
    
    def create_flow_arrows(self):
        """Create arrows emphasizing deterministic flow progression."""
        arrows = []
        
        # Stage 1 flow
        arrows.append(Arrow(LEFT * 3.6, LEFT * 2.4, color=COLOR_SCHEME["stage1"], buff=0.1))
        arrows.append(Arrow(LEFT * 1.6, LEFT * 0.4, color=COLOR_SCHEME["stage1"], buff=0.1))
        
        # Stage 2: Riemannian prior to initial z₀
        arrows.append(Arrow(LEFT * 3.6, LEFT * 2.4, color=COLOR_SCHEME["stage2"], buff=0.1))
        arrows.append(Arrow(LEFT * 1.6, LEFT * 0.4, color=COLOR_SCHEME["stage2"], buff=0.1))
        
        # Stage 3: Deterministic flow progression (key emphasis)
        arrows.append(Arrow(LEFT * 0.6, RIGHT * 1.4, color=COLOR_SCHEME["stage3"], buff=0.1))
        arrows.append(Arrow(RIGHT * 2.6, RIGHT * 3.4, color=COLOR_SCHEME["stage3"], buff=0.1))
        arrows.append(Arrow(RIGHT * 4.6, RIGHT * 5.4, color=COLOR_SCHEME["stage3"], buff=0.1))
        arrows.append(Arrow(RIGHT * 6.6, RIGHT * 7.4, color=COLOR_SCHEME["stage3"], buff=0.1))
        arrows.append(Arrow(RIGHT * 8.6, RIGHT * 9.4, color=COLOR_SCHEME["stage3"], buff=0.1))
        
        # Stage 4: Evaluation flow
        arrows.append(Arrow(LEFT * 1.6, LEFT * 0.4, color=COLOR_SCHEME["stage4"], buff=0.1))
        arrows.append(Arrow(LEFT * 0.6, RIGHT * 1.4, color=COLOR_SCHEME["stage4"], buff=0.1))
        
        return arrows
    
    def create_stage_labels(self):
        """Create stage labels."""
        labels = []
        stage_names = ["Foundation", "Riemannian Prior", "Flow Progression", "Evaluation"]
        positions = [LEFT * 2, LEFT * 1, RIGHT * 1, RIGHT * 2]
        
        for name, pos in zip(stage_names, positions):
            label = Text(name, font_size=7, color=COLOR_SCHEME["primary"])
            label.move_to(pos + UP * 2.5)
            labels.append(label)
        
        return labels
    
    def highlight_flow_progression(self, current_step="z₀"):
        """Highlight the current step in the flow progression."""
        # Reset all blocks
        for block in self.all_blocks.values():
            block.set_opacity(0.3)
            block.set_color(COLOR_SCHEME["text"])
        
        # Highlight flow sequence blocks
        flow_blocks = ["flow_sequence", "z0_to_z1", "z1_to_z2", "zt_minus_1_to_zt"]
        for block_name in flow_blocks:
            if block_name in self.all_blocks:
                self.all_blocks[block_name].set_opacity(0.7)
                self.all_blocks[block_name].set_color(COLOR_SCHEME["flow"])
        
        # Highlight current step
        if current_step == "z₀":
            self.all_blocks["initial_z0"].set_opacity(1.0)
            self.all_blocks["initial_z0"].set_color(COLOR_SCHEME["animation_highlight"])
        elif current_step == "z₁":
            self.all_blocks["z0_to_z1"].set_opacity(1.0)
            self.all_blocks["z0_to_z1"].set_color(COLOR_SCHEME["animation_highlight"])
        elif current_step == "z₂":
            self.all_blocks["z1_to_z2"].set_opacity(1.0)
            self.all_blocks["z1_to_z2"].set_color(COLOR_SCHEME["animation_highlight"])
        elif current_step == "z_T":
            self.all_blocks["zt_minus_1_to_zt"].set_opacity(1.0)
            self.all_blocks["zt_minus_1_to_zt"].set_color(COLOR_SCHEME["animation_highlight"])
    
    def highlight_section(self, section_name, stage=None):
        """Highlight current section and optionally entire stage."""
        # Reset all blocks
        for block in self.all_blocks.values():
            block.set_opacity(0.3)
            block.set_color(COLOR_SCHEME["text"])
        
        # Highlight specific section
        if section_name in self.all_blocks:
            self.all_blocks[section_name].set_opacity(1.0)
            self.all_blocks[section_name].set_color(COLOR_SCHEME["animation_highlight"])
        
        # Highlight entire stage if specified
        if stage:
            stage_blocks = getattr(self, f"stage{stage}_blocks", {})
            for block in stage_blocks.values():
                block.set_opacity(0.7)
                block.set_color(COLOR_SCHEME[f"stage{stage}"])
    
    def position_in_corner(self):
        """Position the scheme in the top-right corner."""
        self.to_corner(UR, buff=0.5)
        self.scale(0.6)
