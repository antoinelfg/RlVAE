"""
Scene 1: VAE Fundamentals & Metric Extraction (Separate Slides)
==============================================================

Renders each slide as a separate final frame for manual presentation creation.
"""

from manim import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.persistent_scheme import PersistentRlVAEScheme
from components.color_scheme import COLOR_SCHEME
from components.animation_helpers import AnimationHelpers
import numpy as np

class VAEMetricExtractionSlide1(Scene):
    """Slide 1: Title slide."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct slide 1."""
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        title = Text("Vanilla VAE — Training Phase", font_size=36, color=COLOR_SCHEME["text"])
        subtitle = Text("Understanding the Architecture", font_size=24, color=COLOR_SCHEME["encoder"])
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.add(title_group)
        self.wait(1)

class VAEMetricExtractionSlide2(Scene):
    """Slide 2: VAE Pipeline."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct slide 2."""
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        # Create all components
        x = Text("x", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(LEFT * 6.5)
        
        encoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["encoder"]).move_to(LEFT * 4)
        enc_lbl = Text("Encoder", font_size=24, color=COLOR_SCHEME["encoder"]).next_to(encoder, UP, buff=0.1)
        enc_text = Text("q(z|x)", color=COLOR_SCHEME["encoder"]).scale(0.8).move_to(encoder.get_center())
        
        latent = RoundedRectangle(corner_radius=0.2, width=3, height=3, color=COLOR_SCHEME["metric"])
        latent_fill = latent.copy().set_fill(color=COLOR_SCHEME["metric"], opacity=0.1)
        lat_lbl = Text("Latent Space", font_size=24, color=COLOR_SCHEME["metric"]).next_to(latent, UP, buff=0.1)
        latent_text = Text("z", color=COLOR_SCHEME["metric"]).scale(1.0).move_to(latent.get_center())
        
        decoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["decoder"]).move_to(RIGHT * 4)
        dec_lbl = Text("Decoder", font_size=24, color=COLOR_SCHEME["decoder"]).next_to(decoder, UP, buff=0.1)
        dec_text = Text("p(x|z)", color=COLOR_SCHEME["decoder"]).scale(0.8).move_to(decoder.get_center())
        
        xhat = Text("x̂", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(RIGHT * 6.5)

        # Arrows
        arr1 = Arrow(x.get_right(), encoder.get_left(), buff=0.1, color=COLOR_SCHEME["encoder"])
        arr2 = Arrow(encoder.get_right(), latent.get_left(), buff=0.1, color=COLOR_SCHEME["metric"])
        arr3 = Arrow(latent.get_right(), decoder.get_left(), buff=0.1, color=COLOR_SCHEME["decoder"])
        arr4 = Arrow(decoder.get_right(), xhat.get_left(), buff=0.1, color=COLOR_SCHEME["highlight"])

        # Add all components
        pipeline = VGroup(
            x, encoder, enc_lbl, enc_text,
            latent, latent_fill, lat_lbl, latent_text,
            decoder, dec_lbl, dec_text,
            xhat, arr1, arr2, arr3, arr4
        )
        
        self.add(pipeline)
        self.wait(1)

class VAEMetricExtractionSlide3(Scene):
    """Slide 3: Metric Extraction Highlight."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct slide 3."""
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        # Create all components (same as slide 2)
        x = Text("x", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(LEFT * 6.5)
        
        encoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["encoder"]).move_to(LEFT * 4)
        enc_lbl = Text("Encoder", font_size=24, color=COLOR_SCHEME["encoder"]).next_to(encoder, UP, buff=0.1)
        enc_text = Text("q(z|x)", color=COLOR_SCHEME["encoder"]).scale(0.8).move_to(encoder.get_center())
        
        latent = RoundedRectangle(corner_radius=0.2, width=3, height=3, color=COLOR_SCHEME["metric"])
        latent_fill = latent.copy().set_fill(color=COLOR_SCHEME["metric"], opacity=0.1)
        lat_lbl = Text("Latent Space", font_size=24, color=COLOR_SCHEME["metric"]).next_to(latent, UP, buff=0.1)
        latent_text = Text("z", color=COLOR_SCHEME["metric"]).scale(1.0).move_to(latent.get_center())
        
        decoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["decoder"]).move_to(RIGHT * 4)
        dec_lbl = Text("Decoder", font_size=24, color=COLOR_SCHEME["decoder"]).next_to(decoder, UP, buff=0.1)
        dec_text = Text("p(x|z)", color=COLOR_SCHEME["decoder"]).scale(0.8).move_to(decoder.get_center())
        
        xhat = Text("x̂", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(RIGHT * 6.5)

        # Arrows
        arr1 = Arrow(x.get_right(), encoder.get_left(), buff=0.1, color=COLOR_SCHEME["encoder"])
        arr2 = Arrow(encoder.get_right(), latent.get_left(), buff=0.1, color=COLOR_SCHEME["metric"])
        arr3 = Arrow(latent.get_right(), decoder.get_left(), buff=0.1, color=COLOR_SCHEME["decoder"])
        arr4 = Arrow(decoder.get_right(), xhat.get_left(), buff=0.1, color=COLOR_SCHEME["highlight"])

        # Add pipeline
        pipeline = VGroup(
            x, encoder, enc_lbl, enc_text,
            latent, latent_fill, lat_lbl, latent_text,
            decoder, dec_lbl, dec_text,
            xhat, arr1, arr2, arr3, arr4
        )
        
        # Add extraction highlight
        extraction_box = SurroundingRectangle(latent, color=COLOR_SCHEME["loss"], buff=0.3)
        extraction_arrow = Arrow(extraction_box.get_bottom(), extraction_box.get_bottom() + DOWN * 1.0, color=COLOR_SCHEME["loss"])
        extraction_label = Text("Extract Metric G(z)", color=COLOR_SCHEME["loss"]).scale(0.7)
        extraction_label.next_to(extraction_arrow, DOWN, buff=0.1)
        
        extraction_group = VGroup(extraction_box, extraction_arrow, extraction_label)
        
        self.add(pipeline, extraction_group)
        self.wait(1)

class VAEMetricExtractionSlide4(Scene):
    """Slide 4: Metric Extraction Process."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct slide 4."""
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        met_title = Text("Metric Extraction via K-means", font_size=36, color=COLOR_SCHEME["text"]).to_edge(UP)
        
        explanation = Text("1. Cluster latent points using K-means\n2. Compute precision matrices M_k\n3. Weight by distance to centroids\n4. Combine into metric tensor G(z)", 
                         font_size=20, color=COLOR_SCHEME["text"], line_spacing=0.8)
        explanation.move_to(ORIGIN)
        
        content_group = VGroup(met_title, explanation)
        
        self.add(content_group)
        self.wait(1)

class VAEMetricExtractionSlide5(Scene):
    """Slide 5: K-means Visualization."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct slide 5."""
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        viz_title = Text("K-means Clustering & Metric Calculation", font_size=36, color=COLOR_SCHEME["text"]).to_edge(UP)
        
        # Create a simple visualization
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"stroke_color": COLOR_SCHEME["text"]}
        ).scale(0.6).move_to(ORIGIN)
        
        # Create some sample points and centroids
        points = VGroup(*[
            Dot(axes.c2p(x, y), color=COLOR_SCHEME["text"], radius=0.05)
            for x, y in [(-2, -1), (-1, 0), (0, 1), (1, 0), (2, -1)]
        ])
        
        centroids = VGroup(*[
            Dot(axes.c2p(x, y), color=COLOR_SCHEME["loss"], radius=0.1)
            for x, y in [(-1.5, -0.5), (1.5, -0.5)]
        ])
        
        viz_group = VGroup(viz_title, axes, points, centroids)
        
        self.add(viz_group)
        self.wait(1)
