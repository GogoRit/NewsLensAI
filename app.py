from src.data_preprocessing import preprocess_articles

# Load and preprocess articles from the CSV
df = preprocess_articles('data/cleaned_articles.csv')

# Display the processed data
print(df[['title', 'cleaned_article', 'ner_entities']].head())

# Save the processed data to a new CSV
df.to_csv('processed_articles.csv', index=False)
