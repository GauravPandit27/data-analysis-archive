import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('loan_data.csv')

# Handling Missing Values
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

# Encoding Categorical Variables
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Feature Scaling
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Creating New Features
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] = scaler.fit_transform(df[['TotalIncome']])
df['IncomeLoanRatio'] = df['TotalIncome'] / df['LoanAmount']
df['IncomeLoanRatio'] = scaler.fit_transform(df[['IncomeLoanRatio']])

# Binning
df['LoanAmount_bin'] = pd.cut(df['LoanAmount'], bins=[0, 100, 200, 700], labels=['Low', 'Medium', 'High'])
df['LoanAmount_bin'] = label_encoder.fit_transform(df['LoanAmount_bin'])

# Interaction Features
df['ApplicantIncome_CoapplicantIncome'] = df['ApplicantIncome'] * df['CoapplicantIncome']

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True)
poly_features = poly.fit_transform(df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']))
df = pd.concat([df, poly_features_df], axis=1)

# Preparing the Final Dataset
X = df.drop(['Loan_Status'], axis=1)
y = df['Loan_Status']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
