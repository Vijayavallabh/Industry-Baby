from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Example text to analyze
text = "Covid cases are increasing fast!"

# Perform sentiment analysis
result = sentiment_analyzer(text)

# Print the result
print(result)
