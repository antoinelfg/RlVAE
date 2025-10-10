"""
Scene 1: VAE Fundamentals & Metric Extraction (Video Version)
============================================================

Regular Manim scene for video rendering, then can be used with manim-slides.
"""

from manim import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.persistent_scheme import PersistentRlVAEScheme
from components.color_scheme import COLOR_SCHEME
from components.animation_helpers import AnimationHelpers
import numpy as np

class VAEMetricExtractionVideo(Scene):
    """Regular VAE scene for video rendering."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Construct the VAE scene for video."""
        
        # Add persistent scheme
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        # Title
        title = Text("Vanilla VAE — Training Phase", font_size=36, color=COLOR_SCHEME["text"])
        subtitle = Text("Understanding the Architecture", font_size=24, color=COLOR_SCHEME["encoder"])
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=2),
            FadeIn(subtitle, shift=UP, run_time=1.5)
        )
        self.wait(2)

        # VAE Pipeline
        self.play(FadeOut(title_group, shift=UP), run_time=1)
        
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

        # Single animation for entire pipeline
        pipeline = VGroup(
            x, encoder, enc_lbl, enc_text,
            latent, latent_fill, lat_lbl, latent_text,
            decoder, dec_lbl, dec_text,
            xhat, arr1, arr2, arr3, arr4
        )
        
        self.play(FadeIn(pipeline, scale=0.8), run_time=3)
        self.wait(2)

        # Metric Extraction Highlight
        extraction_box = SurroundingRectangle(latent, color=COLOR_SCHEME["loss"], buff=0.3)
        extraction_arrow = Arrow(extraction_box.get_bottom(), extraction_box.get_bottom() + DOWN * 1.0, color=COLOR_SCHEME["loss"])
        extraction_label = Text("Extract Metric G(z)", color=COLOR_SCHEME["loss"]).scale(0.7)
        extraction_label.next_to(extraction_arrow, DOWN, buff=0.1)
        
        extraction_group = VGroup(extraction_box, extraction_arrow, extraction_label)
        
        self.play(
            Create(extraction_box, run_time=1),
            Create(extraction_arrow, run_time=1),
            Write(extraction_label, run_time=1)
        )
        self.wait(2)

        # Metric Extraction Process
        self.play(FadeOut(pipeline), FadeOut(extraction_group), run_time=1)
        
        met_title = Text("Metric Extraction via K-means", font_size=36, color=COLOR_SCHEME["text"]).to_edge(UP)
        
        explanation = Text("1. Cluster latent points using K-means\n2. Compute precision matrices M_k\n3. Weight by distance to centroids\n4. Combine into metric tensor G(z)", 
                         font_size=20, color=COLOR_SCHEME["text"], line_spacing=0.8)
        explanation.move_to(ORIGIN)
        
        content_group = VGroup(met_title, explanation)
        
        self.play(
            Write(met_title, run_time=1.5),
            Write(explanation, run_time=2.5)
        )
        self.wait(3)

        # K-means Visualization
        self.play(FadeOut(content_group), run_time=1)
        
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
        
        self.play(
            Write(viz_title, run_time=1),
            Create(axes, run_time=1),
            FadeIn(points, run_time=1),
            FadeIn(centroids, run_time=1)
        )
        
        self.wait(3)
        
        # Final
        self.play(FadeOut(viz_group), run_time=1)
        final_text = Text("Scene 1 Complete!", font_size=36, color=COLOR_SCHEME["accent"])
        self.play(Write(final_text), run_time=2)
        self.wait(2)
