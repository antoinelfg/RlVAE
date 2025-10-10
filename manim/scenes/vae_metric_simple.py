"""
Scene 1: VAE Fundamentals & Metric Extraction (Simplified)
========================================================

Simplified version without complex LaTeX for testing.
"""

from manim import *
from manim_slides import Slide
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.persistent_scheme import PersistentRlVAEScheme
from components.color_scheme import COLOR_SCHEME
from components.animation_helpers import AnimationHelpers
import numpy as np

class VAEMetricExtractionSimple(Slide):
    """Simplified VAE scene for testing."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct the simplified VAE scene."""
        
        # Add persistent scheme
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        # SLIDE 1: Title
        title = Text("Vanilla VAE — Training Phase", font_size=36, color=COLOR_SCHEME["text"])
        subtitle = Text("Understanding the Architecture", font_size=24, color=COLOR_SCHEME["encoder"])
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP, run_time=1)
        )
        self.next_slide()

        # SLIDE 2: Simple VAE Pipeline
        # Input
        x = Text("x", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(LEFT * 6.5)
        
        # Encoder
        encoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["encoder"]).move_to(LEFT * 4)
        enc_lbl = Text("Encoder", font_size=24, color=COLOR_SCHEME["encoder"]).next_to(encoder, UP, buff=0.1)
        enc_text = Text("q(z|x)", color=COLOR_SCHEME["encoder"]).scale(0.8).move_to(encoder.get_center())
        
        # Latent space
        latent = RoundedRectangle(corner_radius=0.2, width=3, height=3, color=COLOR_SCHEME["metric"])
        latent_fill = latent.copy().set_fill(color=COLOR_SCHEME["metric"], opacity=0.1)
        lat_lbl = Text("Latent Space", font_size=24, color=COLOR_SCHEME["metric"]).next_to(latent, UP, buff=0.1)
        latent_text = Text("z", color=COLOR_SCHEME["metric"]).scale(1.0).move_to(latent.get_center())
        
        # Decoder
        decoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["decoder"]).move_to(RIGHT * 4)
        dec_lbl = Text("Decoder", font_size=24, color=COLOR_SCHEME["decoder"]).next_to(decoder, UP, buff=0.1)
        dec_text = Text("p(x|z)", color=COLOR_SCHEME["decoder"]).scale(0.8).move_to(decoder.get_center())
        
        # Output
        xhat = Text("x̂", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(RIGHT * 6.5)

        # Arrows
        arr1 = Arrow(x.get_right(), encoder.get_left(), buff=0.1, color=COLOR_SCHEME["encoder"])
        arr2 = Arrow(encoder.get_right(), latent.get_left(), buff=0.1, color=COLOR_SCHEME["metric"])
        arr3 = Arrow(latent.get_right(), decoder.get_left(), buff=0.1, color=COLOR_SCHEME["decoder"])
        arr4 = Arrow(decoder.get_right(), xhat.get_left(), buff=0.1, color=COLOR_SCHEME["highlight"])

        # Animate components sequentially
        self.play(FadeOut(title_group, shift=UP))
        
        # Input
        self.play(Write(x), run_time=0.5)
        
        # Encoder
        self.play(
            Create(VGroup(encoder, enc_lbl)),
            Write(enc_text),
            run_time=0.5
        )
        
        # Arrow 1
        self.play(Create(arr1), run_time=0.5)

        # Latent space
        self.play(
            Create(latent),
            FadeIn(latent_fill),
            Write(lat_lbl),
            Write(latent_text),
            run_time=0.5
        )
        
        # Arrow 2
        self.play(Create(arr2), run_time=0.5)

        # Decoder
        self.play(
            Create(VGroup(decoder, dec_lbl)),
            Write(dec_text),
            run_time=0.5
        )
        
        # Arrow 3
        self.play(Create(arr3), run_time=0.5)
        
        # Arrow 4 and output
        self.play(
            Create(arr4),
            Write(xhat),
            run_time=0.5
        )

        self.next_slide()

        # SLIDE 3: Metric Extraction Highlight
        # Highlight latent space
        extraction_box = SurroundingRectangle(latent, color=COLOR_SCHEME["loss"], buff=0.3)
        extraction_arrow = Arrow(extraction_box.get_bottom(), extraction_box.get_bottom() + DOWN * 1.0, color=COLOR_SCHEME["loss"])
        extraction_label = Text("Extract Metric G(z)", color=COLOR_SCHEME["loss"]).scale(0.7)
        extraction_label.next_to(extraction_arrow, DOWN, buff=0.1)
        
        self.play(
            Create(extraction_box),
            Create(extraction_arrow),
            Write(extraction_label),
            run_time=1
        )

        self.next_slide()

        # SLIDE 4: K-means Visualization
        # Clean up previous elements
        vae_elements = VGroup(
            x, encoder, enc_lbl, enc_text,
            latent, latent_fill, lat_lbl, latent_text,
            decoder, dec_lbl, dec_text,
            xhat, arr1, arr2, arr3, arr4,
            extraction_box, extraction_arrow, extraction_label
        )
        
        self.play(FadeOut(vae_elements), run_time=0.5)
        
        # Title for metric extraction
        met_title = Text("Metric Extraction via K-means", font_size=36, color=COLOR_SCHEME["text"]).to_edge(UP)
        self.play(Write(met_title), run_time=1)

        # Simple explanation
        explanation = Text("1. Cluster latent points using K-means\n2. Compute precision matrices M_k\n3. Weight by distance to centroids\n4. Combine into metric tensor G(z)", 
                         font_size=20, color=COLOR_SCHEME["text"], line_spacing=0.8)
        explanation.move_to(ORIGIN)
        
        self.play(Write(explanation), run_time=2)
        self.wait(1)
        
        # Clean up
        self.play(
            FadeOut(met_title),
            FadeOut(explanation),
            run_time=0.5
        )

        self.next_slide()
        self.wait(1)
