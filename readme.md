# Fake News Detection

A machine learning project that classifies news articles as **Fake** or **Real**.
It uses a TF-IDF vectorizer together with a trained scikit-learn classifier to
predict whether a given piece of text (optionally with its title) is fake news.

## Features

- Predict a single news text with `detect_one_text`.
- Predict a batch of news texts with `detect`.
- Optional combining of a news `title` and `text` before prediction.
- Colored, human-readable output of predictions.

## Project Structure

| File | Description |
| --- | --- |
| `detection.py` | Main `detection` class that loads the model and runs predictions. |
| `utils.py` | Helper functions to load the vectorizer and merge title with text. |
| `model.pkl` | Pre-trained classification model. |
| `Vectorizer.pkl` | Pre-trained TF-IDF vectorizer. |
| `Fake News Detection.ipynb` | Notebook used for data exploration and model training. |

## Installation

```bash
git clone https://github.com/meysam-kazemi/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt
```

## Usage

Run the built-in demo:

```bash
python detection.py
```

Or use the `detection` class in your own code:

```python
from detection import detection

det = detection()

# Single text
det.detect_one_text("Some news article text...", with_title=False)

# Multiple texts
det.detect(
    texts=["First article...", "Second article..."],
    with_title=False,
    verbose=True,
)
```

## How It Works

1. The TF-IDF vectorizer transforms the input text into numerical features.
2. The trained model predicts the class for each input.
3. Predictions are mapped to labels: `0 -> Real`, `1 -> Fake`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
