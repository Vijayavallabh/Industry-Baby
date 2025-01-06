import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.le = LabelEncoder()
        
    def process_features(self, df):
        # Numerical features
        num_cols = ['Annual_Income', 'Num_Bank_Accounts',
                   'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
                   'Delay_from_due_date', 'Num_of_Delayed_Payment',
                   'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                   'Total_EMI_per_month', 'Amount_invested_monthly',
                   'Monthly_Balance']
                   
        # Categorical features  
        cat_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
        
        # Scale numerical features
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        # Encode categorical features
        for col in cat_cols:
            df[col] = self.le.fit_transform(df[col])
            
        return df