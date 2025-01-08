from prompts import QUERY_EXPANSION_PROMPT, RESPONSE_PROMPT


def expand_query(client, query, model="llama-3.3-70b-versatile"):
    """ Expand the user query based on the original query """
    
    chat = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": QUERY_EXPANSION_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Original Query: {query}",
                },
            ],
            model=model,
            temperature = 0.7,
    )
    
    return chat.choices[0].message.content


def response_to_original_query(client, user_query, retrieved_docs, model="llama-3.3-70b-versatile"):
    """
    Generate a response to the original query using the retrieved documents as context.
    """
    context = "\n".join(retrieved_docs)
    
    prompt = RESPONSE_PROMPT.format(context=context, user_query=user_query)

    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses context to answer queries accurately."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.5,
    )
    
    return response.choices[0].message.content