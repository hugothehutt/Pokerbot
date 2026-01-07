"""
Strategy persistence for saving and loading trained models.

Supports:
- CFR trained strategies
- Future reinforcement learning models
- Checkpointing during long training runs
- Model versioning and metadata
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json
import pickle
import gzip
import time
import os


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    model_type: str  # "cfr", "rl", "neural", etc.
    iterations: int
    training_time_seconds: float
    created_at: float
    version: str = "1.0"
    config: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        return cls(**data)


class StrategyPersistence:
    """
    Handles saving and loading trained poker strategies.

    Supports multiple formats:
    - JSON: Human-readable, good for CFR strategies
    - Pickle: Fast, good for complex objects
    - Compressed: For large models

    Example usage:
        # Save CFR strategy
        persistence = StrategyPersistence("./models")
        persistence.save_cfr(trainer, "my_cfr_v1", iterations=100000)

        # Load later
        trainer = persistence.load_cfr("my_cfr_v1")

        # Save RL model (future)
        persistence.save_model(rl_agent, "rl_agent_v1", model_type="rl")
    """

    def __init__(self, base_dir: str = "./poker_models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_cfr(
        self,
        trainer,  # CFRTrainer instance
        name: str,
        description: str = "",
        compress: bool = False,
    ) -> Path:
        """
        Save a CFR trainer's strategy.

        Args:
            trainer: CFRTrainer instance
            name: Model name (used as filename)
            description: Optional description
            compress: Whether to gzip compress

        Returns:
            Path to saved model directory
        """
        model_dir = self.base_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = ModelMetadata(
            model_type="cfr",
            iterations=trainer.iterations_trained,
            training_time_seconds=0,  # Could track this
            created_at=time.time(),
            config={
                "stack_size": trainer.solver.stack_size,
                "big_blind": trainer.solver.big_blind,
            },
            metrics={
                "exploitability": trainer.solver.strategy.get_strategy(0).get_exploitability(),
                "info_sets": len(trainer.solver.strategy.get_strategy(0)),
            },
            description=description,
        )

        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Save strategies
        for player_idx in range(2):
            strategy = trainer.solver.strategy.get_strategy(player_idx)
            strategy_data = {
                key: info_set.to_dict()
                for key, info_set in strategy.info_sets.items()
            }

            filepath = model_dir / f"player_{player_idx}.json"
            if compress:
                filepath = filepath.with_suffix(".json.gz")
                with gzip.open(filepath, "wt") as f:
                    json.dump(strategy_data, f)
            else:
                with open(filepath, "w") as f:
                    json.dump(strategy_data, f, indent=2)

        print(f"Saved CFR model to: {model_dir}")
        return model_dir

    def load_cfr(self, name: str):
        """
        Load a saved CFR strategy.

        Args:
            name: Model name

        Returns:
            CFRTrainer with loaded strategy
        """
        from pokerbot.strategy.cfr import CFRTrainer
        from pokerbot.strategy.cfr.strategy import InfoSetData

        model_dir = self.base_dir / name

        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        # Create trainer with same config
        trainer = CFRTrainer(
            stack_size=metadata.config.get("stack_size", 100.0),
            big_blind=metadata.config.get("big_blind", 1.0),
        )
        trainer.iterations_trained = metadata.iterations

        # Load strategies
        for player_idx in range(2):
            filepath = model_dir / f"player_{player_idx}.json"
            compressed_path = model_dir / f"player_{player_idx}.json.gz"

            if compressed_path.exists():
                with gzip.open(compressed_path, "rt") as f:
                    strategy_data = json.load(f)
            elif filepath.exists():
                with open(filepath, "r") as f:
                    strategy_data = json.load(f)
            else:
                continue

            strategy = trainer.solver.strategy.get_strategy(player_idx)
            for key, info_data in strategy_data.items():
                strategy.info_sets[key] = InfoSetData.from_dict(info_data)

        print(f"Loaded CFR model: {name} ({metadata.iterations} iterations)")
        return trainer

    def save_model(
        self,
        model: Any,
        name: str,
        model_type: str,
        metadata: Optional[ModelMetadata] = None,
        compress: bool = True,
    ) -> Path:
        """
        Save any model using pickle (for RL, neural networks, etc.).

        Args:
            model: The model object to save
            name: Model name
            model_type: Type identifier ("rl", "neural", "hybrid")
            metadata: Optional metadata
            compress: Whether to compress

        Returns:
            Path to saved model
        """
        model_dir = self.base_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create default metadata if not provided
        if metadata is None:
            metadata = ModelMetadata(
                model_type=model_type,
                iterations=getattr(model, "iterations_trained", 0),
                training_time_seconds=getattr(model, "training_time", 0),
                created_at=time.time(),
            )

        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Save model
        model_path = model_dir / "model.pkl"
        if compress:
            model_path = model_path.with_suffix(".pkl.gz")
            with gzip.open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        print(f"Saved {model_type} model to: {model_dir}")
        return model_dir

    def load_model(self, name: str) -> Any:
        """
        Load a pickled model.

        Args:
            name: Model name

        Returns:
            The loaded model object
        """
        model_dir = self.base_dir / name

        # Try compressed first
        compressed_path = model_dir / "model.pkl.gz"
        regular_path = model_dir / "model.pkl"

        if compressed_path.exists():
            with gzip.open(compressed_path, "rb") as f:
                model = pickle.load(f)
        elif regular_path.exists():
            with open(regular_path, "rb") as f:
                model = pickle.load(f)
        else:
            raise FileNotFoundError(f"No model found at {model_dir}")

        # Load metadata for info
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = ModelMetadata.from_dict(json.load(f))
            print(f"Loaded {metadata.model_type} model: {name}")

        return model

    def list_models(self) -> list[dict]:
        """List all saved models with their metadata."""
        models = []
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    metadata["name"] = model_dir.name
                    models.append(metadata)
        return models

    def delete_model(self, name: str) -> bool:
        """Delete a saved model."""
        import shutil
        model_dir = self.base_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"Deleted model: {name}")
            return True
        return False


# Convenience functions
_default_persistence = None


def get_persistence(base_dir: str = "./poker_models") -> StrategyPersistence:
    """Get the default persistence instance."""
    global _default_persistence
    if _default_persistence is None or str(_default_persistence.base_dir) != base_dir:
        _default_persistence = StrategyPersistence(base_dir)
    return _default_persistence


def save_trained_strategy(trainer, name: str, **kwargs) -> Path:
    """Quick save for CFR trainer."""
    return get_persistence().save_cfr(trainer, name, **kwargs)


def load_trained_strategy(name: str):
    """Quick load for CFR trainer."""
    return get_persistence().load_cfr(name)
