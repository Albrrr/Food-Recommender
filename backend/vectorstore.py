#FAISS vector store
#creates vectors of each menu item to create a vector space of the items

import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vectorstore(json_path="DATA.json"):
    with open(json_path) as f:
        data = json.load(f)

    documents = []
    for restaurant_name, restaurant_data in data.items():
        for category, items in restaurant_data.items():
            if isinstance(items, dict):
                for item_name, item_details in items.items():
                    if isinstance(item_details, dict) and 'description' in item_details:
                        text = f"Restaurant: {restaurant_name}, Category: {category}, Item: {item_name}, Description: {item_details['description']}"
                    elif isinstance(item_details, str):
                        text = f"Restaurant: {restaurant_name}, Category: {category}, Item: {item_name}, Description: {item_details}"
                    else:
                        continue
                    documents.append(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(documents, embeddings)
