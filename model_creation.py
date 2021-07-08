# Importing the necessary libraries.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# vars
SPECIES_MAP = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
SEX_MAP = {'Male': 0, 'Female': 1}
ISLAND_MAP = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map(SPECIES_MAP)

# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map(SEX_MAP)

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map(ISLAND_MAP)

# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

print(pd.Series(rf_clf.predict(X_test)).value_counts())
