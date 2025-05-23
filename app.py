from ingest import load_cases_from_csv, chunk_documents

csv_docs = load_cases_from_csv('./data/legal_text_classification.csv')

# Now you can chunk them with your existing chunk_documents function
nodes = chunk_documents(csv_docs)