from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import json
import argparse
from tqdm import tqdm
import logging
import sys
import time
from datetime import datetime, timedelta

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def log_with_timestamp(message):
    """Add timestamp to log messages"""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")

def load_model(model_name="meta-llama/Llama-2-7b-hf"):
    """Load the model and tokenizer with memory optimizations and progress updates."""
    start_time = time.time()
    log_with_timestamp("Starting tokenizer download and initialization...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_time = time.time() - start_time
    log_with_timestamp(f"Tokenizer loaded in {tokenizer_time:.1f} seconds")
    
    log_with_timestamp("Starting model download and initialization...")
    log_with_timestamp("This may take several minutes - downloading approximately 13GB of model files...")
    
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    total_time = time.time() - start_time
    log_with_timestamp(f"Model fully loaded in {total_time:.1f} seconds")
    return model, tokenizer

def create_prompt(tweet: str, date: str) -> str:
    """Create a classification prompt for a single tweet."""
    return f"""Pretend that you are a classifier that predicts whether a post has political content or not. We define political tweets as content related to e>

Please classify the following text, denoted in <text>, as 'political' or 'not political'. If the text is not political, return "unclear".
Return in a JSON format in the following way:
{{
    "political": <two values, 'political' or 'not political'>,
    "reason_political": <optional, a 1 sentence reason for why the text is political. If none, return an empty string, "">,
}}

<date>
{date}
</date>
<text>
{tweet}
</text>

Classification:"""
def classify_tweet(model, tokenizer, device, tweet: str, date: str, max_length: int = 768) -> dict:
    """Classify a single tweet."""
    prompt = create_prompt(tweet, date)
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        response_text = response[response.find("{"):response.rfind("}")+1]
        classification = json.loads(response_text)
        
        if classification["political"] not in ["political", "not political"]:
            classification["political"] = "unclear"
        if "reason_political" not in classification:
            classification["reason_political"] = ""
            
        return classification
    except:
	return {
            "political": "unclear",
            "reason_political": ""
        }


def process_tweets(input_file: str, output_file: str, batch_size: int = 5):
   """Process all tweets with enhanced progress monitoring."""
    try:
        log_with_timestamp(f"Loading tweets from {input_file}")
        with open(input_file, 'r') as f:
            tweets = json.load(f)
        
       	total_tweets = len(tweets)
        log_with_timestamp(f"Found {total_tweets} tweets to process")
        
       	model, tokenizer, device = load_model()
        
       	results = []
        start_time = time.time()
        
       	for i, tweet_data in tqdm(enumerate(tweets), total=total_tweets, unit="tweet"):
            try:
                tweet_text = tweet_data['text']
                tweet_date = tweet_data['date']
                
               	classification = classify_tweet(model, tokenizer, device, tweet_text, tweet_date)
                
               	results.append({
                    'date': tweet_date,
                    'text': tweet_text,
                    'political': classification['political'],
                    'reason': classification['reason_political']
                })
                
               	if (i + 1) % batch_size == 0:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    log_with_timestamp(f"Progress saved - {i+1} tweets processed")
                
            except Exception as e:
                log_with_timestamp(f"Error processing tweet {i}: {str(e)}")
                continue
        
        # Save final results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        total_time = time.time() - start_time
        avg_time_per_tweet = total_time / total_tweets
        
        log_with_timestamp(f"\nProcessing complete!")
        log_with_timestamp(f"Total time: {total_time:.1f} seconds")
        log_with_timestamp(f"Average time per tweet: {avg_time_per_tweet:.1f} seconds")
        
        # Classification summary
        classifications = [r['political'] for r in results]
        log_with_timestamp("\nClassification Summary:")
        log_with_timestamp(f"Political: {classifications.count('political')}")
        log_with_timestamp(f"Not Political: {classifications.count('not political')}")
        log_with_timestamp(f"Unclear: {classifications.count('unclear')}")
        
    except Exception as e:
        log_with_timestamp(f"Fatal error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Classify tweets for political content using LLaMA-2-7B')
    parser.add_argument('--input', required=True, help='Input JSON file containing tweets')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    parser.add_argument('--batch-size', type=int, default=5, help='How often to save progress')
    args = parser.parse_args()
    
    process_tweets(args.input, args.output, args.batch_size)

if __name__ == "__main__":
    main()
