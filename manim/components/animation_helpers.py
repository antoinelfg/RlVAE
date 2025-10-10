"""
RlVAE Animation Helpers
=======================

Reusable animation functions for the RlVAE presentation.
"""

from manim import *
from .color_scheme import COLOR_SCHEME

class AnimationHelpers:
    """Helper class for common animations in RlVAE presentation."""
    
    @staticmethod
    def highlight_with_pulse(mobject, duration=1.0):
        """Create a pulsing highlight animation."""
        return Succession(
            mobject.animate.set_color(COLOR_SCHEME["animation_highlight"]).scale(1.1),
            mobject.animate.scale(1/1.1),
            run_time=duration
        )
    
    @staticmethod
    def fade_in_sequence(mobjects, lag_ratio=0.1):
        """Fade in a sequence of mobjects."""
        return LaggedStart(
            *[FadeIn(mob) for mob in mobjects],
            lag_ratio=lag_ratio
        )
    
    @staticmethod
    def create_progress_bar(progress, width=4, height=0.2):
        """Create a progress bar animation."""
        background = Rectangle(
            width=width, height=height,
            fill_color=COLOR_SCHEME["background"],
            fill_opacity=0.3,
            stroke_color=COLOR_SCHEME["text"]
        )
        
        progress_bar = Rectangle(
            width=width * progress, height=height,
            fill_color=COLOR_SCHEME["animation_progress"],
            fill_opacity=0.8,
            stroke_width=0
        ).align_to(background, LEFT)
        
        return VGroup(background, progress_bar)
    
    @staticmethod
    def animate_progress_update(progress_bar, new_progress, duration=1.0):
        """Animate progress bar update."""
        return progress_bar[1].animate.become(
            Rectangle(
                width=progress_bar[0].width * new_progress,
                height=progress_bar[0].height,
                fill_color=COLOR_SCHEME["animation_progress"],
                fill_opacity=0.8,
                stroke_width=0
            ).align_to(progress_bar[0], LEFT)
        )
    
    @staticmethod
    def create_flow_arrow(start, end, color=None):
        """Create a flow arrow with consistent styling."""
        if color is None:
            color = COLOR_SCHEME["flow"]
        
        return Arrow(
            start, end,
            color=color,
            buff=0.1,
            max_tip_length_to_length_ratio=0.15,
            stroke_width=2
        )
    
    @staticmethod
    def create_metric_ellipse(center, scale=1.0, angle=None):
        """Create a metric tensor ellipse."""
        ellipse = Ellipse(
            width=0.4 * scale,
            height=0.2 * scale,
            color=COLOR_SCHEME["metric"],
            fill_opacity=0.1,
            stroke_opacity=0.3
        )
        ellipse.move_to(center)
        
        if angle is not None:
            ellipse.rotate(angle)
        
        return ellipse
    
    @staticmethod
    def animate_metric_evolution(ellipses, duration=2.0):
        """Animate metric tensor evolution."""
        animations = []
        for i, ellipse in enumerate(ellipses):
            animations.append(
                ellipse.animate.scale(1 + 0.2 * np.sin(i * 0.5))
                .set_color(COLOR_SCHEME["animation_highlight"])
            )
        
        return LaggedStart(*animations, lag_ratio=0.1, run_time=duration)
    
    @staticmethod
    def create_equation_highlight(equation, part_indices, color=None):
        """Highlight specific parts of an equation."""
        if color is None:
            color = COLOR_SCHEME["animation_highlight"]
        
        highlights = []
        for indices in part_indices:
            if isinstance(indices, tuple):
                start, end = indices
                highlight = SurroundingRectangle(
                    equation[start:end],
                    color=color,
                    buff=0.1
                )
                highlights.append(highlight)
        
        return highlights
    
    @staticmethod
    def animate_equation_build(equation, duration=1.0):
        """Animate equation building character by character."""
        return AddTextLetterByLetter(equation, run_time=duration)
    
    @staticmethod
    def create_manifold_visualization(manifold_func, t_range, color=None):
        """Create a manifold visualization."""
        if color is None:
            color = COLOR_SCHEME["metric"]
        
        manifold = ParametricFunction(
            manifold_func,
            t_range=t_range,
            color=color,
            fill_opacity=0.2
        )
        
        return manifold
    
    @staticmethod
    def animate_flow_progression(points, arrows, duration=1.0):
        """Animate flow progression through points."""
        animations = []
        
        # Animate points appearing
        for point in points:
            animations.append(FadeIn(point))
        
        # Animate arrows appearing
        for arrow in arrows:
            animations.append(Create(arrow))
        
        return LaggedStart(*animations, lag_ratio=0.2, run_time=duration)
    
    @staticmethod
    def create_loss_curve(epochs, losses, color=None):
        """Create a loss curve visualization."""
        if color is None:
            color = COLOR_SCHEME["loss"]
        
        # Create axes
        axes = Axes(
            x_range=[0, max(epochs), max(epochs)//5],
            y_range=[min(losses), max(losses), (max(losses) - min(losses))/5],
            axis_config={"stroke_color": COLOR_SCHEME["text"]}
        )
        
        # Create curve
        curve = axes.plot_line_graph(
            x_values=epochs,
            y_values=losses,
            line_color=color,
            add_vertex_dots=True
        )
        
        return VGroup(axes, curve)
    
    @staticmethod
    def animate_loss_convergence(loss_curve, duration=2.0):
        """Animate loss convergence."""
        return Create(loss_curve, run_time=duration)
