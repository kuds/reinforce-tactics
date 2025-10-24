"""
Experiment tracking with Weights & Biases and TensorBoard
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np


class ExperimentTracker:
    """
    Unified experiment tracking interface.
    Supports both W&B and TensorBoard.
    """
    
    def __init__(
        self,
        project_name: str = "reinforcetactics",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: str = "./logs",
        use_wandb: bool = True,
        use_tensorboard: bool = True
    ):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: Name of project
            experiment_name: Name of this experiment
            config: Configuration dictionary
            log_dir: Directory for logs
            use_wandb: Whether to use W&B
            use_tensorboard: Whether to use TensorBoard
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{os.getpid()}"
        self.config = config or {}
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config
                )
                print("✅ Weights & Biases initialized")
            except ImportError:
                print("⚠️  wandb not installed, skipping")
        
        # Initialize TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))
                print("✅ TensorBoard initialized")
            except ImportError:
                print("⚠️  tensorboard not installed, skipping")
        
        # Save config
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log scalar metrics."""
        # W&B
        if self.wandb:
            self.wandb.log(metrics, step=step)
        
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """Log histogram."""
        # W&B
        if self.wandb:
            self.wandb.log({name: self.wandb.Histogram(values)}, step=step)
        
        # TensorBoard
        if self.writer:
            self.writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: np.ndarray, step: int):
        """Log image."""
        # W&B
        if self.wandb:
            self.wandb.log({name: self.wandb.Image(image)}, step=step)
        
        # TensorBoard
        if self.writer:
            # Assuming image is (H, W, C)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            image = image.transpose(2, 0, 1)  # (C, H, W)
            self.writer.add_image(name, image, step)
    
    def log_video(self, name: str, video: np.ndarray, step: int, fps: int = 4):
        """Log video."""
        # W&B
        if self.wandb:
            self.wandb.log({name: self.wandb.Video(video, fps=fps)}, step=step)
        
        # TensorBoard
        if self.writer:
            # Assuming video is (T, H, W, C)
            video = video.transpose(0, 3, 1, 2)  # (T, C, H, W)
            video = video[np.newaxis, ...]  # (1, T, C, H, W)
            self.writer.add_video(name, video, step, fps=fps)
    
    def log_table(self, name: str, data: Dict[str, list]):
        """Log table/dataframe."""
        # W&B
        if self.wandb:
            import pandas as pd
            df = pd.DataFrame(data)
            self.wandb.log({name: self.wandb.Table(dataframe=df)})
    
    def save_model(self, model, name: str):
        """Save model checkpoint."""
        model_path = self.log_dir / f"{name}.pt"
        
        # Save locally
        import torch
        torch.save(model.state_dict(), model_path)
        
        # Upload to W&B
        if self.wandb:
            self.wandb.save(str(model_path))
    
    def finish(self):
        """Finish experiment tracking."""
        if self.wandb:
            self.wandb.finish()
        
        if self.writer:
            self.writer.close()
        
        print(f"✅ Experiment {self.experiment_name} finished")


# Convenience function
def create_tracker(
    experiment_name: str,
    config: Dict[str, Any],
    **kwargs
) -> ExperimentTracker:
    """Create experiment tracker with defaults."""
    return ExperimentTracker(
        experiment_name=experiment_name,
        config=config,
        **kwargs
    )
