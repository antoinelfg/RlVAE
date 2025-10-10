class GUGUSReconstructionPipeline(Slide):
    def __init__(self):
        super().__init__()
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
            "distribution": ORANGE,
            "encoder": "#2196F3",  # Blue
            "decoder": "#FF9800",  # Orange
        }

    def construct(self):
        # SLIDE 1: Title
        title = Text("GUGUS Reconstruction Pipeline", font_size=36)
        subtitle = Text("From Data to Reconstruction", font_size=24, color=BLUE)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP, run_time=1)
        )
        self.next_slide()

        # SLIDE 2: Data Manifold
        self.show_data_manifold()
        self.next_slide()

        # SLIDE 3: Encoding Process
        self.show_encoding_process()
        self.next_slide()

        # SLIDE 4: Diffusion Process
        self.show_diffusion_process()
        self.next_slide()

        # SLIDE 5: Decoding Process
        self.show_decoding_process()
        self.next_slide()

        # SLIDE 6: Final Comparison
        self.show_final_comparison()
        self.next_slide()

class GUGUSTrainingPipeline(Slide):
    def __init__(self):
        super().__init__()
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
            "distribution": ORANGE,
            "encoder": "#2196F3",  # Blue
            "decoder": "#FF9800",  # Orange
        }

    def construct(self):
        # SLIDE 1: Title
        title = Text("GUGUS Training Pipeline", font_size=36)
        subtitle = Text("Learning the Manifold Structure", font_size=24, color=BLUE)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)
        
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP, run_time=1)
        )
        self.next_slide()

        # SLIDE 2: Initial Setup
        self.show_initial_setup()
        self.next_slide()

        # SLIDE 3: Training Process
        self.show_training_process()
        self.next_slide()

        # SLIDE 4: Evolution
        self.show_evolution()
        self.next_slide()

        # SLIDE 5: Final Results
        self.show_final_results()
        self.next_slide() 