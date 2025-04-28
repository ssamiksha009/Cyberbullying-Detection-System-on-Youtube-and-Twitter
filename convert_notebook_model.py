import os
import json
import torch
import nbformat
from nbconvert import PythonExporter
import re
from transformers import AutoModel, AutoTokenizer

def extract_model_from_notebook(notebook_path, output_model_path, output_tokenizer_path):
    """
    Extract the trained BERTweet model from the Jupyter notebook and save it.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_model_path: Path where the model will be saved
        output_tokenizer_path: Path where the tokenizer will be saved
    """
    print(f"Converting notebook {notebook_path} to model at {output_model_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = f.read()
    
    notebook = nbformat.reads(notebook_content, as_version=4)
    
    # Extract the Python code from the notebook
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(notebook)
    
    # Define the BERTweet model class
    class BERTweetClassifier(torch.nn.Module):
        def __init__(self, bertweet_model, num_classes=5):  # Updated for 5 classes
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

    try:
        # Initialize base BERTweet model and tokenizer
        bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        
        # Create classifier model
        model = BERTweetClassifier(bertweet)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), output_model_path)
        print(f"Model saved to {output_model_path}")
        
        # Save tokenizer
        tokenizer.save_pretrained(os.path.dirname(output_tokenizer_path))
        print(f"Tokenizer saved to {output_tokenizer_path}")
        
    except Exception as e:
        print(f"Error extracting model: {str(e)}")
        
    return model, tokenizer


def main():
    # Update this to match your notebook's path
    notebook_path = "FINAL_MODEL.ipynb"
    output_model_path = "models/bertweet_cyberbullying_model.pth"
    output_tokenizer_path = "models/bertweet_tokenizer"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Extract and save model
    model, tokenizer = extract_model_from_notebook(
        notebook_path, 
        output_model_path, 
        output_tokenizer_path
    )

if __name__ == "__main__":
    main()          