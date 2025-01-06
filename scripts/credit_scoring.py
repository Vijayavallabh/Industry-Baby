from scripts.train import train_model
from scripts.predict import predict_credit_score

def main():
    # Train model using training data
    train_model(r"datasets\Creditscore_train_cleaned.csv")
    
    # Generate predictions on test data
    results_df, predictions, probabilities = predict_credit_score(r"datasets\Creditscore_test_cleaned.csv")
    
    # Save predictions 
    results_df.to_csv('credit_score_predictions.csv', index=False)
    print("\nPredictions saved to 'credit_score_predictions.csv'")

if __name__ == "__main__":
    main()