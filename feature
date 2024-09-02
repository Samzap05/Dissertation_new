from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words='english')

# Fit and transform the text data into feature vectors
X = vectorizer.fit_transform(df['text'])

# Convert to DataFrame for readability
features_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(features_df.head())  # Display the first few rows of the feature DataFrame
