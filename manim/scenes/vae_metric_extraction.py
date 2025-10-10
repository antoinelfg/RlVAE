"""
Scene 1: VAE Fundamentals & Metric Extraction
============================================

This scene covers the VAE basics and metric extraction process.
Extracted and adapted from the existing full_manim_pipeline.py
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

class VAEMetricExtraction(Slide):
    """Scene 1: VAE Fundamentals & Metric Extraction."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def flash_animation(self, mobject):
        """Create a flash animation effect."""
        return Succession(
            mobject.animate.set_color(COLOR_SCHEME["animation_highlight"]),
            mobject.animate.set_color(mobject.get_color()),
        )
    
    def construct(self):
        """Construct the VAE and metric extraction scene."""
        
        # Add persistent scheme
        self.add(self.scheme)
        self.scheme.highlight_section("vanilla_vae", stage=1)
        
        # SLIDE 1: Title with dynamic entrance
        title = Text("Vanilla VAE — Training Phase", font_size=36, color=COLOR_SCHEME["text"])
        subtitle = Text("Understanding the Architecture", font_size=24, color=COLOR_SCHEME["encoder"])
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP, run_time=1)
        )
        self.next_slide()

        # SLIDE 2: Full Pipeline with dynamic build-up
        # Initialize components with initial opacity 0
        x = MathTex(r"x_j^{(i)}", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(LEFT * 6.5)
        encoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["encoder"]).move_to(LEFT * 4)
        enc_lbl = Text("Encoder", font_size=24, color=COLOR_SCHEME["encoder"]).next_to(encoder, UP, buff=0.1)
        enc1 = MathTex(r"q_{\phi}(z|x)", color=COLOR_SCHEME["encoder"]).scale(0.6).move_to(encoder.get_center() + UP * 0.3)
        enc2 = MathTex(r"\mathcal N(\mu_{\phi}(x_j^{(i)}), \sigma_{\phi}(x_j^{(i)}))", color=COLOR_SCHEME["encoder"]).scale(0.5).next_to(enc1, DOWN, buff=0.2)
        
        # Create latent space with gradient fill
        latent = RoundedRectangle(corner_radius=0.2, width=3, height=3, color=COLOR_SCHEME["metric"])
        latent_fill = latent.copy().set_fill(color=COLOR_SCHEME["metric"], opacity=0.1)
        latent_group = VGroup(latent, latent_fill).move_to(ORIGIN)
        lat_lbl = Text("Latent Space", font_size=24, color=COLOR_SCHEME["metric"]).next_to(latent, UP, buff=0.1)
        
        decoder = Rectangle(width=3, height=1.5, color=COLOR_SCHEME["decoder"]).move_to(RIGHT * 4)
        dec_lbl = Text("Decoder", font_size=24, color=COLOR_SCHEME["decoder"]).next_to(decoder, UP, buff=0.1)
        dec1 = MathTex(r"p_{\theta}(z|x)", color=COLOR_SCHEME["decoder"]).scale(0.6).move_to(decoder.get_center() + UP * 0.3)
        dec2 = MathTex(r"\mathcal N(\mu_{\theta}(z_j^{(i)}), diag(\sigma_{\theta}^2(z_j^{(i)})))", color=COLOR_SCHEME["decoder"]).scale(0.45).next_to(dec1, DOWN, buff=0.2)
        xhat = MathTex(r"\hat x_j^{(i)}", color=COLOR_SCHEME["highlight"]).scale(1.2).move_to(RIGHT * 6.5)

        # Create arrows with gradient
        arr1 = Arrow(x.get_right(), encoder.get_left(), buff=0.1, color=COLOR_SCHEME["encoder"])
        arr2 = Arrow(encoder.get_right(), latent.get_left(), buff=0.1, color=COLOR_SCHEME["metric"])
        arr3 = Arrow(latent.get_right(), decoder.get_left(), buff=0.1, color=COLOR_SCHEME["decoder"])
        arr4 = Arrow(decoder.get_right(), xhat.get_left(), buff=0.1, color=COLOR_SCHEME["highlight"])

        # Prior information with dynamic appearance
        prior_lbl = Text("Prior:", font_size=24, color=COLOR_SCHEME["text"]).move_to(latent.get_center())
        zij = MathTex(r"z_j^{(i)}", color=COLOR_SCHEME["metric"]).scale(0.9).next_to(prior_lbl, UP, buff=0.2)
        prior = MathTex(r"p(z)=\mathcal N(0,I)", color=COLOR_SCHEME["metric"]).scale(0.8).next_to(prior_lbl, DOWN, buff=0.2)

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
        extraction_box1 = SurroundingRectangle(latgroup, color=COLOR_SCHEME["loss"], buff=0.3)
        jac_arrow1 = Arrow(extraction_box1.get_bottom(), extraction_box1.get_bottom() + DOWN * 1.0, color=COLOR_SCHEME["loss"])
        jac_label1 = MathTex(r"c_k \;=\;\frac{1}{|\mathcal{I}_k|}\sum_{(i,j)\in\mathcal{I}_k} z_j^{(i)}", color=COLOR_SCHEME["loss"]).scale(0.7)
        
        # Add glow effect to the box
        glow = extraction_box1.copy().set_stroke(color=COLOR_SCHEME["loss"], opacity=0.5, width=10)
        
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

        centro = Text("Centroids via k-means:", font_size=24, color=COLOR_SCHEME["loss"])
        self.play(
            Write(centro.next_to(extraction_box1, UP, buff=0.1)),
            jac_label1.animate.set_color(COLOR_SCHEME["loss"]),
            run_time=0.5
        )

        # Decoder highlight with similar effects
        decgroup = VGroup(decoder, dec_lbl)
        extraction_box2 = SurroundingRectangle(decgroup, color=COLOR_SCHEME["highlight"], buff=0.3)
        glow2 = extraction_box2.copy().set_stroke(color=COLOR_SCHEME["highlight"], opacity=0.5, width=10)
        
        self.play(
            Create(extraction_box2),
            FadeIn(glow2, rate_func=there_and_back),
            run_time=0.5
        )

        jac_arrow2 = Arrow(extraction_box2.get_bottom(), extraction_box2.get_bottom() + DOWN * 1.0, color=COLOR_SCHEME["highlight"])
        jac_label2 = MathTex(r"J_\theta(z)", color=COLOR_SCHEME["highlight"]).scale(0.7)
        
        self.play(
            Create(jac_arrow2),
            Write(jac_label2.next_to(jac_arrow2, DOWN, buff=0.1)),
            run_time=0.5
        )

        jaco = Text("Access to Jacobians:", font_size=24, color=COLOR_SCHEME["highlight"])
        self.play(
            Write(jaco.next_to(extraction_box2, UP, buff=0.1)),
            run_time=0.5
        )

        # Final equation with dynamic build-up
        eqm = MathTex(r"M_k =J_\theta(c_k)^{\!T}\,J_\theta(c_k)", color=COLOR_SCHEME["highlight"]).scale(0.7)
        self.play(
            Write(eqm.next_to(jac_label2, DOWN, buff=0.1)),
            run_time=0.5
        )

        self.next_slide()

        # SLIDE 4: Metric Extraction
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
        
        met_title = Text("Metric Extraction", font_size=36, color=COLOR_SCHEME["text"]).to_edge(UP)
        self.play(Write(met_title), run_time=1)

        # Centroid equation with dynamic build-up
        jac_label1 = MathTex(r"c_k \;=\;\frac{1}{|\mathcal{I}_k|}\sum_{(i,j)\in\mathcal{I}_k} z_j^{(i)}", color=COLOR_SCHEME["loss"]).scale(0.7)
        jac_label1.move_to(UP * 2 + LEFT * 5)
        self.play(Write(jac_label1), run_time=0.5)

        # Precision matrix equation with highlight
        eqm = MathTex(r"M_k =J_\theta(c_k)^{\!T}\,J_\theta(c_k)", color=COLOR_SCHEME["highlight"]).scale(0.7)
        eqm.next_to(jac_label1, DOWN, buff=0.5)
        self.play(Write(eqm), run_time=0.5)

        # Add inverse covariance with flash effect
        eqm2 = MathTex(r" =\Sigma_k^{-1}\; }", color=COLOR_SCHEME["highlight"]).scale(0.7)
        eqm2.next_to(eqm, RIGHT, buff=0.1)
        self.play(
            Write(eqm2),
            Flash(eqm2, color=COLOR_SCHEME["highlight"], line_length=0.2),
            run_time=0.5
        )
        eqmgroup = VGroup(eqm, eqm2)

        # Weight equation with sequential animation
        wk = MathTex(
            r"w_k(z_{j}^{(i)})\;=\;\frac{e^{-\frac{\|z_{j}^{(i)} - c_k\|^2}{2\,\lambda\,T}}}"
            r"{\displaystyle \sum_{\ell=1}^{K} e^{-\frac{\|z_{j}^{(i)} - c_{\ell}\|^2}{2\,\lambda\,T}}}",
            color=COLOR_SCHEME["metric"]
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
        centroids = Text("Centroids obtained via k-means", font_size=24, color=COLOR_SCHEME["text"])
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
            color=COLOR_SCHEME["text"],
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
            color=COLOR_SCHEME["text"],
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
            tex_to_color_map={r"\lambda": COLOR_SCHEME["metric"], "bandwidth": COLOR_SCHEME["metric"]}
        ).scale(0.5)
        lambda_def.next_to(wk, DOWN, buff=0.4).align_to(wk, LEFT)
        self.play(
            Write(lambda_def),
            Flash(lambda_def[0], color=COLOR_SCHEME["metric"], line_length=0.2),
            run_time=0.5
        )

        T_def = MathTex(
            r"T > 0:",
            r"\text{ softmax temperature}",
            tex_to_color_map={r"T": COLOR_SCHEME["metric"], "temperature": COLOR_SCHEME["metric"]}
        ).scale(0.5)
        T_def.next_to(lambda_def, DOWN, buff=0.2).align_to(lambda_def, LEFT)
        self.play(
            Write(T_def),
            Flash(T_def[0], color=COLOR_SCHEME["metric"], line_length=0.2),
            run_time=0.5
        )

        # Final equation with box animation
        eq = MathTex(
            r"\boxed{"
            r"G^{-1}(z_{j}^{(i)}) \;=\; \sum_{k=1}^K w_k(z_{j}^{(i)}) \, M_k \;\Longrightarrow\;G(z_{j}^{(i)})\approx\bigl(G^{-1}(z_{j}^{(i)})+\varepsilon I\bigr)^{-1}\!"
            r"}",
            tex_to_color_map={
                r"G^{-1}(z_{i}^j)": COLOR_SCHEME["text"],
                r"G(z_{i}^j)": COLOR_SCHEME["text"],
                r"w_k(z_{i}^j)": COLOR_SCHEME["text"],
                r"M_k": COLOR_SCHEME["text"],
                r"\Longrightarrow": COLOR_SCHEME["text"],
                r"\sum": COLOR_SCHEME["text"],
            }
        ).scale(0.6).move_to(RIGHT * 2.4 + DOWN * 2.6)
        
        box = SurroundingRectangle(eq, buff=0.2)
        self.play(Write(eq), run_time=1)
        self.play(Create(box), run_time=0.5)
        self.play(FadeOut(box), run_time=0.5)

        self.next_slide()
        self.wait(1)
        
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

        # SLIDE 5: K-means Visualization
        title = Text("K-means Clustering and Metric Calculation", font_size=36, color=COLOR_SCHEME["text"])
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Create axes for latent space with grid
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            axis_config={
                "stroke_color": COLOR_SCHEME["grid"],
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
                color=COLOR_SCHEME["grid"],
                stroke_opacity=0.3,
                stroke_width=0.5
            )
            grid.add(line)
        
        # Horizontal lines
        for y in y_range:
            line = Line(
                start=axes.c2p(-4, y),
                end=axes.c2p(4, y),
                color=COLOR_SCHEME["grid"],
                stroke_opacity=0.3,
                stroke_width=0.5
            )
            grid.add(line)

        self.play(Create(axes), Create(grid))
        self.wait(1)

        # Create prior distribution visualization - only 3σ circle
        prior_circle = Circle(radius=3 * 0.8, color=COLOR_SCHEME["encoder"], fill_opacity=0.05)
        prior_circle.move_to(axes.get_center())
        prior_label = MathTex(r"p(z) = \mathcal{N}(0, I)", color=COLOR_SCHEME["encoder"]).scale(0.6)
        prior_label.next_to(prior_circle, UP, buff=0.1)
        
        self.play(Create(prior_circle), Write(prior_label))
        self.wait(1)

        # Explanation of sampling
        sampling_text = Text("Sampling points from N(0,1)", font_size=24, color=COLOR_SCHEME["text"])
        sampling_text.move_to(RIGHT * 3 + UP * 2.5)
        self.play(Write(sampling_text))
        self.wait(1)

        # Generate random points from N(0,1)
        np.random.seed(42)  # For reproducibility
        n_points = 100
        points = np.random.normal(0, 1, (n_points, 2))
        
        # Create dots for points
        dots = VGroup()
        for i, (x, y) in enumerate(points):
            dot = Dot(axes.c2p(x, y), color=COLOR_SCHEME["text"], radius=0.06)
            dots.add(dot)

        self.play(
            LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.02),
            run_time=2
        )
        self.wait(1)

        # K-means clustering explanation
        kmeans_text = Text("K-means Clustering Process", font_size=24, color=COLOR_SCHEME["text"])
        kmeans_text.move_to(RIGHT * 3 + UP * 2.5)
        self.play(
            Transform(sampling_text, kmeans_text)
        )
        self.wait(1)

        # K-means clustering animation
        K = 3
        centroids = points[np.random.choice(n_points, K, replace=False)]
        centroid_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=COLOR_SCHEME["loss"], radius=0.12)
            for x, y in centroids
        ])
        
        # Add centroid labels
        centroid_labels = VGroup()
        for i, (x, y) in enumerate(centroids):
            label = MathTex(f"c_{i+1}", color=COLOR_SCHEME["loss"]).scale(0.5)
            label.next_to(centroid_dots[i], UP, buff=0.1)
            centroid_labels.add(label)
        
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in centroid_dots], lag_ratio=0.1),
            LaggedStart(*[Write(label) for label in centroid_labels], lag_ratio=0.1)
        )
        self.wait(1)

        # Animate k-means iterations
        cluster_colors = [COLOR_SCHEME["loss"], COLOR_SCHEME["metric"], COLOR_SCHEME["decoder"]]
        for iteration in range(3):
            # Assign points to clusters
            distances = np.array([[np.linalg.norm(p - c) for c in centroids] for p in points])
            clusters = np.argmin(distances, axis=1)
            
            # Create cluster lines
            cluster_lines = VGroup()
            for i, (x, y) in enumerate(points):
                c_x, c_y = centroids[clusters[i]]
                dist = np.linalg.norm([x - c_x, y - c_y])
                opacity = max(0.1, min(0.5, 1.0 - dist/4))
                line = Line(
                    axes.c2p(x, y),
                    axes.c2p(c_x, c_y),
                    color=cluster_colors[clusters[i]],
                    stroke_opacity=opacity
                )
                cluster_lines.add(line)
            
            self.play(Create(cluster_lines), run_time=0.5)
            self.wait(0.5)
            
            # Update centroids
            if iteration < 2:  # Don't update on last iteration
                new_centroids = []
                for k in range(K):
                    cluster_points = points[clusters == k]
                    if len(cluster_points) > 0:
                        new_centroid = np.mean(cluster_points, axis=0)
                        new_centroids.append(new_centroid)
                    else:
                        new_centroids.append(centroids[k])
                
                # Animate centroid movement
                centroid_animations = []
                for i, (old_pos, new_pos) in enumerate(zip(centroids, new_centroids)):
                    centroid_animations.append(
                        centroid_dots[i].animate.move_to(axes.c2p(new_pos[0], new_pos[1]))
                    )
                    centroid_animations.append(
                        centroid_labels[i].animate.next_to(centroid_dots[i], UP, buff=0.1)
                    )
                
                self.play(*centroid_animations, run_time=0.5)
                centroids = np.array(new_centroids)
                self.wait(0.5)
            
            # Remove lines for next iteration
            if iteration < 2:
                self.play(FadeOut(cluster_lines), run_time=0.3)

        # Final result text
        final_text = Text("Metric extracted from cluster centroids", font_size=24, color=COLOR_SCHEME["text"])
        final_text.move_to(RIGHT * 3 + DOWN * 2.5)
        self.play(Write(final_text))
        self.wait(1)
        
        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(grid),
            FadeOut(prior_circle),
            FadeOut(prior_label),
            FadeOut(sampling_text),
            FadeOut(dots),
            FadeOut(centroid_dots),
            FadeOut(centroid_labels),
            FadeOut(cluster_lines),
            FadeOut(final_text),
            run_time=1
        )

        self.next_slide()
        self.wait(1)
