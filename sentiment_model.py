import json
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


print("Model labels:", config.id2label)

tweets = []
with open("tweets-2020-2021-subset-rnd.jl", "r") as file:
    for i, line in enumerate(file):
        if i >= 20:
            break
        tweet = json.loads(line)
        tweets.append(tweet["text"])

# Perform sentiment analysis on each tweet and save results to CSV
with open("sentiment_analysis_results.csv", "w", newline='') as csvfile:
    fieldnames = ['Sentiment', 'Positive_Prob', 'Neutral_Prob', 'Negative_Prob']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, text in enumerate(tweets):
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment = config.id2label[np.argmax(scores)]
        
        writer.writerow({
            'Sentiment': sentiment,
            'Positive_Prob': np.round(float(scores[config.label2id['positive']]), 4),
            'Neutral_Prob': np.round(float(scores[config.label2id['neutral']]), 4),
            'Negative_Prob': np.round(float(scores[config.label2id['negative']]), 4)
        })

print("Sentiment analysis results saved to 'sentiment_analysis_results.csv'")