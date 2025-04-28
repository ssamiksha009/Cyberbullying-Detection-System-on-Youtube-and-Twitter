import os
import torch
from transformers import AutoModel, AutoTokenizer

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Define the same model architecture as in your notebook
class BERTweetClassifier(torch.nn.Module):
    def __init__(self, bertweet_model, num_classes=2):
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

# Load the pre-trained BERTweet model
print("Loading BERTweet base model...")
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

# Create your classifier model
print("Creating cyberbullying classifier model...")
model = BERTweetClassifier(bertweet)

# Method 1: If you have the weights available locally
weights_path = input("Enter path to your model weights (leave empty if not available): ").strip()
if weights_path and os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
else:
    print("No weights loaded. Creating a fresh model.")
    
    # You can add basic initialization here
    # This will create a model with the right architecture but without your trained weights
    print("Initializing with default weights.")
    # The model will have random initialization for the classifier layer

# Save the model
output_path = 'models/bertweet_cyberbullying_model.pth'
torch.save(model.state_dict(), output_path)
print(f"Model saved to {output_path}")

# Also save the tokenizer for convenience
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
tokenizer.save_pretrained('models/bertweet_tokenizer')
print("Tokenizer saved to models/bertweet_tokenizer")

print("\nTest the model with a sample prediction:")
print("Loading the saved model...")
test_model = BERTweetClassifier(bertweet)
test_model.load_state_dict(torch.load(output_path, map_location=torch.device('cpu')))
test_model.eval()

# Test with sample text
sample_text = "This is a test comment"
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')

with torch.no_grad():
    outputs = test_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    predictions = torch.softmax(outputs, dim=1)
    bullying_score = predictions[0][1].item()

print(f"Sample text: '{sample_text}'")
print(f"Bullying score: {bullying_score:.4f}")
print(f"Classification: {'Bullying' if bullying_score > 0.5 else 'Not bullying'}")