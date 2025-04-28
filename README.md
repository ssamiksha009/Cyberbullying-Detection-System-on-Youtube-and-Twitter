# Cyberbullying Detection System

## Overview
This advanced Cyberbullying Detection System uses BERTweet, a state-of-the-art transformer model, to analyze comments from YouTube and Twitter for various forms of cyberbullying.

## Features
- Real-time comment analysis
- Detection of 5 cyberbullying classes:
  1. Sexual Harassment
  2. Cyberstalking
  3. Doxing
  4. Privacy Violations
  5. Slut Shaming
- Interactive web interface
- Visualizations of cyberbullying statistics
- Support for YouTube and Twitter platforms

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cyberbullying-detection.git
cd cyberbullying-detection
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file in the project root with your API keys:
```
YOUTUBE_API_KEY=your_youtube_api_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret
```

### 5. Convert Trained Model
Place your trained Jupyter Notebook in the project directory and run:
```bash
python convert_notebook_model.py
```

### 6. Run the Application
```bash
python app.py
```

Open a web browser and navigate to `http://localhost:5000`

## Project Structure
- `app.py`: Main Flask application
- `convert_notebook_model.py`: Script to convert Jupyter Notebook model
- `requirements.txt`: Project dependencies
- `static/index.html`: Frontend interface
- `models/`: Directory for storing trained model and tokenizer

## Model Training
The model was trained on a custom dataset with 5 cyberbullying classes using the BERTweet base model. Achieved 86% accuracy in classification.

## Limitations
- API rate limits for YouTube and Twitter
- Model accuracy depends on training data
- Requires continuous model retraining and improvement

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify your license here]

## Contact
[Your contact information]