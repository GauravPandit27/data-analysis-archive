import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Step 2: Understanding the Dataset
print("Initial dataset info:\n", df.info())
print("\nInitial dataset statistics:\n", df.describe())

# Step 3: Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Step 4: Encoding Categorical Variables
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 5: Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1  # Initialize to 1 (true)
df['IsAlone'].loc[df['FamilySize'] > 1] = 0  # Set to 0 (false) if FamilySize > 1
df.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch'], inplace=True)

# Step 6: Normalization
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'FamilySize']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 7: Splitting the Dataset
target = 'Survived'
X = df.drop(columns=[target])
y = df[target]

# Display the final preprocessed dataset
print("\nFinal preprocessed features:\n", X.head())
print("\nFinal target variable:\n", y.head())

# Visualization
# 1. Distribution of Age
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Distribution of Fare
plt.figure(figsize=(10, 5))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# 3. Survival rate by Sex
plt.figure(figsize=(10, 5))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex (1=Male, 0=Female)')
plt.ylabel('Survival Rate')
plt.show()

# 4. Survival rate by Embarked
plt.figure(figsize=(10, 5))
sns.barplot(x='Embarked_Q', y='Survived', data=df)
plt.title('Survival Rate by Embarked (Q)')
plt.xlabel('Embarked Q (1=Yes, 0=No)')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Embarked_S', y='Survived', data=df)
plt.title('Survival Rate by Embarked (S)')
plt.xlabel('Embarked S (1=Yes, 0=No)')
plt.ylabel('Survival Rate')
plt.show()

# 5. Survival rate by FamilySize
plt.figure(figsize=(10, 5))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.show()

# 6. Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.2)
plt.title('Correlation Matrix')
plt.show()
