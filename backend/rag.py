#RAG logic
#Makes call to the AI model to search the RAG for fitting documents
#After processing the data, generates a response by making another call to the AI

def run_rag(query, vector_store, model, tokenizer, k=3):
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Use the retrieved restaurant menu "
                "information from the RAG databased to answer the user's question. "
                "Your job is provide food recommendations that are closest to the "
                "food that the user likes. Using the prefereneces outlines by the user "
                "and the food items available in the database, come up with three items "
                "that the user would like. For each item, give a paragraph for why the item "
                "fits the taste prefereneces of the user and any dietary information if the "
                "menu provides it. Ensure that you include the name of the restaurant with each "
                "dish. Keep the paragraph concise and use a friendly tone."
            )
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}"
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("assistant")[-1].strip()
