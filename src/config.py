from pathlib import Path

# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

# Serialized artifacts
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "Vectorizer.pkl"

# Label mapping used by the classifier
CLASSES = {1.0: "Fake", 0.0: "Real"}
