# Streamlit Credit Score Prediction App

This project is a Streamlit application designed to predict credit scores based on user-uploaded CSV data and to scrape and analyze tweets. The application processes the data through cleaning and explainability steps, providing model predictions and SHAP values for interpretability.

## Project Structure

```
streamlit-app
├── src
│   ├── app.py              # Main entry point for the Streamlit application
│   ├── data_clean.py       # Functions for cleaning input CSV data
│   ├── explainability.py   # Logic for model training, prediction, and SHAP values
│   ├── llm_explain.py      # Functionality for generating recommendations using a language model
│   ├── tweet.py            # Functions for scraping and analyzing tweets
├── requirements.txt        # List of dependencies required for the project
└── README.md               # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd streamlit-app
   ```

2. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```
   streamlit run src/app.py
   ```

## Usage Guidelines

- **Credit Score Prediction:**
  - Upload a CSV file containing the necessary data for credit score prediction.
  - The application will clean the data, process it through the model, and display the predicted credit scores along with SHAP values for interpretability.

- **Scrape and Analyze Tweets:**
  - Enter a Twitter username and the number of tweets to fetch.
  - The application will scrape the tweets and provide recommendations and evaluations based on the input tweet.

## Application Functionality

- **Data Cleaning:** The application handles duplicates, imputes missing values, and formats the data for analysis.
- **Model Training and Prediction:** It utilizes ensemble models to predict credit scores based on the cleaned data.
- **SHAP Values:** The application generates SHAP values to explain the model's predictions, providing insights into feature importance.
- **Tweet Scraping and Analysis:** The application scrapes tweets from a specified user and provides recommendations and evaluations for a given tweet.

## License

This project is licensed under the MIT License.