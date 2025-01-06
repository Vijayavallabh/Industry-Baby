from scripts.data_clean import clean_data
from scripts.predict import predict_credit_score

def main():
    input_file = "datasets/test.csv"
    
    # Clean and preprocess the data
    df = clean_data(input_file)
    print("Data cleaned and preprocessed.")
    
    # Get predictions using the trained model
    results_df, predictions, probabilities = predict_credit_score(df)
    
    # Save predictions to CSV
    output_file = "credit_score_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")

if __name__ == "__main__":
    main()