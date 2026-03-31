import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Student_Performance_new.csv")

# Convert final_grade into pass/fail
def grade_to_pass_fail(grade):
    if grade in ['A', 'B', 'C']:
        return 'Pass'
    else:
        return 'Fail'

df['pass_fail'] = df['final_grade'].apply(grade_to_pass_fail)

# Drop columns not needed
df = df.drop(['student_id', 'final_grade'], axis=1)

# Encode categorical columns
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split features and target
X = df.drop('pass_fail', axis=1)
y = df['pass_fail']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))