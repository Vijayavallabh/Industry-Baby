import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_clean import *
from explainability import *
from llm_explain import *
import joblib
from tweet import *
import seaborn as sns


def credit_score_prediction():
    st.title("Credit Score Prediction and Explainability")

    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stTitle {
        color: #4CAF50;
        font-size: 2.5em;
        font-weight: bold;
    }
    .stSubheader {
        color: #4CAF50;
        font-size: 1.5em;
        font-weight: bold;
    }
    .stDataFrame {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .stMarkdown {
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(data)

        cleaned_data = clean_data(data)
        
        st.subheader("Predicting Credit Scores...")
        preds, probs, cleaned_data = predict_credit_score(cleaned_data)

        results_df = pd.DataFrame({
            'Predicted_Score': preds,
            'Good_Prob': probs[:, 0],
            'Standard_Prob': probs[:, 1],
            'Poor_Prob': probs[:, 2]
        })
        st.subheader("Prediction Results")
        st.dataframe(results_df)

        st.subheader("Distribution of Predictions")
        for i, row in results_df.iterrows():
            st.write(f"Data Point {i+1}")
            fig, ax = plt.subplots(figsize=(4, 4))  # Smaller figure size
            ax.pie([row['Good_Prob'], row['Standard_Prob'], row['Poor_Prob']], labels=['Good', 'Standard', 'Poor'], autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            ax.axis('equal')
            st.pyplot(fig)
        
        st.subheader("Generating SHAP Values...")
        with st.container():
            import shap
            shap_values = generate_shap_values(cleaned_data)
            shap.summary_plot([shap_values.values[:, :, i] for i in range(shap_values.shape[2])], cleaned_data, plot_size=(8, 6))  # Smaller plot size
            fig = plt.gcf()
            st.pyplot(fig)

def scrape_and_analyze_tweets():
    st.title("Scrape and Analyze Tweets")

    tabs = st.tabs(["Scrape Tweets", "Tweet Recommender"])

    with tabs[0]:
        st.subheader("Scrape Tweets")
        username = st.text_input("Enter the Twitter username:")
        tweet_limit = st.number_input("Enter the number of tweets to fetch:", min_value=1, max_value=100, value=10)
        
        if st.button("Scrape Tweets"):
            if username:
                with st.spinner("Scraping tweets..."):
                    tweets = scrape_tweets(username, tweet_limit)
                    if tweets:
                        st.subheader(f"Fetched {len(tweets)} tweets for user @{username}")
                        for idx, tweet in enumerate(tweets, start=1):
                            st.write(f"{idx}: {tweet}")
                    else:
                        st.error("No tweets found or an error occurred.")
            else:
                st.error("Please enter a Twitter username.")

    with tabs[1]:
        st.subheader("Tweet Recommender")
        tweet = st.text_area("Enter the tweet for analysis:")
        if st.button("Analyze Tweet"):
            if tweet:
                analysis = recommend(tweet)
                st.subheader("Recommendations")
                st.write(analysis['recommendation'])
                st.subheader("Evaluation")
                st.write(analysis['evaluation'])
            else:
                st.error("Please enter a tweet for analysis.")

def main():
    st.set_page_config(page_title="Credit Score Prediction and Explainability", page_icon=":bar_chart:", layout="wide")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Credit Score Prediction", "Scrape and Analyze Tweets"])

    # Display the selected page
    if page == "Credit Score Prediction":
        credit_score_prediction()
    elif page == "Scrape and Analyze Tweets":
        scrape_and_analyze_tweets()

if __name__ == "__main__":
    main()