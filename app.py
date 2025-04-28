import os
import re
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModel, AutoTokenizer
from googleapiclient.discovery import build
import tweepy
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Cyberbullying Classification Mapping
CYBERBULLYING_CLASSES = [
    'Sexual Harassment', 
    'Cyberstalking', 
    'Doxing', 
    'Privacy Violations', 
    'Slut Shaming'
]

# BERTweet Classifier Model
class BERTweetClassifier(torch.nn.Module):
    def __init__(self, bertweet_model, num_classes=5):
        super(BERTweetClassifier, self).__init__()
        self.bertweet = bertweet_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bertweet.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_model_and_tokenizer():
    """Load pre-trained BERTweet model and tokenizer"""
    try:
        # Load base BERTweet model
        bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        
        # Initialize model
        model = BERTweetClassifier(bertweet)
        
        # Load trained weights
        model_path = os.path.join('models', 'bertweet_cyberbullying_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and tokenizer at startup
MODEL, TOKENIZER = load_model_and_tokenizer()

def preprocess_text(text):
    """Preprocess text for model input"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|\#', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_cyberbullying_class(text):
    """Enhanced cyberbullying prediction with keyword and explicit detection"""
    if MODEL is None or TOKENIZER is None:
        logger.error("Model not initialized")
        return None

    try:
        # Preprocessing
        processed_text = preprocess_text(text.lower())
        
        # Explicit keyword lists for different cyberbullying classes
        explicit_keywords = {
            'Sexual Harassment': [
                'sexual', 'inappropriate', 'perverted', 'sexting', 
                'inappropriate touch', 'sexual comment'
            ],
            'Cyberstalking': [
                'stalk', 'following', 'harass', 'threatening', 
                'constantly messaging', 'watching'
            ],
            'Doxing': [
                'personal info', 'address', 'phone', 'private details', 
                'leaked', 'doxxing'
            ],
            'Privacy Violations': [
                'private pic', 'revenge porn', 'leaked', 'private info', 
                'intimate photo'
            ],
            'Slut Shaming': [
                'slut', 'whore', 'easy', 'promiscuous', 'slutty', 
                'cheap', 'loose'
            ]
        }

        # Offensive language detection
        offensive_words = [
            'fuck', 'fucking', 'shit', 'bitch', 'asshole', 
            'damn', 'cunt', 'dick', 'cock', 'pussy'
        ]

        # Check for explicit keywords and offensive language
        detected_classes = []
        for cls, keywords in explicit_keywords.items():
            if any(keyword in processed_text for keyword in keywords):
                detected_classes.append(cls)
        
        is_offensive = any(word in processed_text for word in offensive_words)

        # ML Model Prediction
        inputs = TOKENIZER(processed_text, return_tensors="pt", 
                           truncation=True, max_length=128, padding='max_length')
        
        with torch.no_grad():
            outputs = MODEL(input_ids=inputs["input_ids"], 
                            attention_mask=inputs["attention_mask"])
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top class and its probability
            top_prob, top_class = torch.max(probabilities, dim=1)
            
            # Combine ML prediction with keyword detection
            is_bullying = (
                top_prob.item() > 0.5 or  # ML model confidence
                is_offensive or  # Offensive language
                len(detected_classes) > 0  # Explicit keywords
            )
            
            # Prioritize detected classes or use ML prediction
            primary_class = (
                detected_classes[0] if detected_classes 
                else CYBERBULLYING_CLASSES[top_class.item()]
            )
            
            return {
                'class': primary_class,
                'probability': top_prob.item(),
                'is_bullying': is_bullying,
                'detected_classes': detected_classes
            }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None

# Setup API Clients (load from environment variables)
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'your_youtube_api_key')
TWITTER_API_KEY = os.environ.get('TWITTER_API_KEY', 'your_twitter_api_key')
TWITTER_API_SECRET = os.environ.get('TWITTER_API_SECRET', 'your_twitter_api_secret')
TWITTER_ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN', 'your_twitter_access_token')
TWITTER_ACCESS_SECRET = os.environ.get('TWITTER_ACCESS_SECRET', 'your_twitter_access_secret')
TWITTER_BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN', 'your_twitter_bearer_token')

# Initialize YouTube API Client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize Twitter API Client
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
twitter_api = tweepy.API(auth)
# Initialize Twitter/X API Client
try:
    client = tweepy.Client(
        bearer_token=TWITTER_BEARER_TOKEN,
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET
    )
except Exception as e:
    logging.error(f"Failed to initialize Twitter API client: {str(e)}")
    client = None

def get_youtube_comments(video_id, max_results=500):
    """Fetch YouTube comments with cyberbullying classification"""
    try:
        comments = []
        next_page_token = None
        
        while len(comments) < max_results:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token
            ).execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comment_text = comment['textDisplay']
                
                # Predict cyberbullying
                prediction = predict_cyberbullying_class(comment_text)
                
                comments.append({
                    'id': item['id'],
                    'text': comment_text,
                    'author': comment['authorDisplayName'],
                    'timestamp': comment['publishedAt'],
                    'isBullying': prediction.get('is_bullying', False),
                    'bullyingClass': prediction.get('class', 'Unknown'),
                    'bullyingProbability': prediction.get('probability', 0),
                    'detectedClasses': prediction.get('detected_classes', [])
                })
                
            
            # Check for more comments
            next_page_token = response.get('nextPageToken')
            if not next_page_token or len(comments) >= max_results:
                break
        
        return comments
    except Exception as e:
        logger.error(f"YouTube comment fetch error: {str(e)}")
        return []

def get_twitter_comments(tweet_id, max_results=100):
    """Fetch Twitter comments with enhanced cyberbullying classification"""
    if client is None:
        logger.error("Twitter API client is not initialized")
        return []

    try:
        # Validate tweet ID
        if not tweet_id or not tweet_id.isdigit():
            logger.error(f"Invalid tweet ID: {tweet_id}")
            return []

        # Ensure max_results is within API limits
        max_results = max(10, min(max_results, 100))

        # Fetch the tweet first
        try:
            tweet = client.get_tweet(tweet_id, expansions=['author_id'])
        except Exception as e:
            logger.error(f"Could not fetch tweet with ID: {tweet_id}. Error: {str(e)}")
            return []

        if not tweet or not tweet.data:
            logger.error(f"No tweet found with ID: {tweet_id}")
            return []

        # Search for replies
        try:
            # Ensure we have a username to search
            if not tweet.includes or 'users' not in tweet.includes:
                logger.error("Could not find tweet author")
                return []

            # Implement exponential backoff for rate limiting
            comments = []
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    reply_search = client.search_recent_tweets(
                        query=f'to:{tweet.includes["users"][0].username}',
                        max_results=max_results,
                        tweet_fields=['text', 'author_id', 'created_at'],
                        expansions=['author_id']
                    )
                    
                    # Process replies
                    if reply_search.data:
                        for reply in reply_search.data:
                            # Predict cyberbullying
                            prediction = predict_cyberbullying_class(reply.text)
                            
                            comments.append({
                                'id': str(reply.id),
                                'text': reply.text,
                                'author': str(reply.author_id),  # User ID as string
                                'timestamp': reply.created_at.isoformat(),
                                'isBullying': prediction.get('is_bullying', False),
                                'bullyingClass': prediction.get('class', 'Unknown'),
                                'bullyingProbability': prediction.get('probability', 0),
                                'detectedClasses': prediction.get('detected_classes', [])
                            })
                    
                    # If successful, break the retry loop
                    break
                
                except tweepy.TooManyRequests:
                    # Implement exponential backoff
                    wait_time = (2 ** retry_count) * 10  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                
                except Exception as e:
                    logger.error(f"Error searching for replies (attempt {retry_count + 1}): {str(e)}")
                    retry_count += 1
            
            return comments

        except Exception as e:
            logger.error(f"Error searching for replies: {str(e)}")
            return []

    except Exception as e:
        logger.error(f"Unexpected Twitter comment fetch error: {str(e)}")
        return []

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/analyze-comments', methods=['POST'])
def analyze_comments():
    """API endpoint to analyze comments from YouTube or Twitter"""
    data = request.json
    platform = data.get('platform')
    content_id = data.get('contentId')
    
    if not platform or not content_id:
        return jsonify({"error": "Missing platform or content ID"}), 400
    
    try:
        comments = []
        
        if platform == 'youtube':
            comments = get_youtube_comments(content_id)
        elif platform == 'twitter':
            comments = get_twitter_comments(content_id)
        else:
            return jsonify({"error": "Invalid platform"}), 400
        
        return jsonify({"comments": comments})
    except Exception as e:
        logger.error(f"Error analyzing comments: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))