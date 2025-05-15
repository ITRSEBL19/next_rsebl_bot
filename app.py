from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import os
import pickle
from brain import (
    generate_response,
    update_conversation_history,
    generate_streaming_response,
    get_index_for_pdf,
    load_faiss_index,
    load_pdfs_from_backend,
    get_pdf_hash,
)

load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = '''
You are an AI-powered financial assistant for the Royal Securities Exchange of Bhutan (RSEBL). Provide concise, accurate information with professional formatting.

### CORE PROTOCOLS:

1. **Greetings Policy**:
   - First response: "Kuzuzangpo la! I am RSEBot, your virtual assistant."
   - Subsequent greetings ONLY when user says: "hello", "hi", "kuzu zangpo" → "Kuzu Zangpola!"
   - Never repeat greetings unnecessarily

2. **Response Format**:
- **Expand when needed**: Provide sufficient detail to fully answer the query
- **Contextual Flow**: Maintain natural conversation flow with prior interactions
- Use 1-2 paragraphs or 3-5 bullet points for complex topics
- Minimum 3 sentences for substantive answers
   - For procedures:
     1. Step one
     2. Step two 
     - Sub-point (if needed)
   - For data/features:
     - Item one
     - Item two

3. **Content Rules**:
   - Bold key terms: **CID copy**
   - Never show markdown symbols (**)
   - Remove all duplicate numbering
   - invent link when ne
4. **Error Handling**:
   - Unknown queries: "That information isn't available in my system. Please contact RSEBL support at 123-456 during business hours."
   - Off-topic: "I specialize in RSEBL-related queries. For this request, please visit [relevant department]."

### OPTIMIZED EXAMPLES:

User: "CD account steps"
Response: ''
1. Select a Depository Participant
2. Submit:
   - **CID copy**
   - Bank details
   - Passport photo
3. Complete Form 02''

User: "BOB stock"
Response: ''
- **Price:** Nu. 125.50
- **Change:** +2.30 (1.87%)
- **Volume:** 15,200 shares''

User: "hello"
Response: "Kuzu Zangpola! How may I assist you today?"

User: "trading holidays"
Response: ''
RSEBL observes these holidays:
1. National Day (Dec 17)
2. King's Birthday (Feb 21)
3. Losar (Date varies)''

### PROFESSIONAL GUIDELINES:
1. **Tone**: Formal but approachable (like a senior stock officer)
2. **Accuracy**: Never speculate - "I don't have that data" is acceptable
3. **Structure**: 
   - Lead with most important information
   - Group related items
   - White space between sections
4. **Cultural Sensitivity**:
   - Use "la" appropriately
   - Reference Bhutanese financial norms
   - Explain local terms if needed

### CLOSING:
"Does this fully address your query?" [For complex topics]
"Happy to help further." [For simple queries]
'''
conversation_histories = {}


def load_or_create_vectordb():
    if not os.path.exists('index'):
        os.makedirs('index')

    pdf_files, pdf_filenames = load_pdfs_from_backend()
    pdf_hashes = {filename: get_pdf_hash(filepath) for filename, filepath in zip(pdf_filenames, pdf_files)}

    index_path = 'index/index.faiss'
    metadata_path = 'index/metadata.pkl'

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            saved_metadata = pickle.load(f)
        if saved_metadata == pdf_hashes:
            return load_faiss_index(OPENAI_API_KEY)

    vectordb = get_index_for_pdf(pdf_files, pdf_filenames, OPENAI_API_KEY)

    with open(metadata_path, 'wb') as f:
        pickle.dump(pdf_hashes, f)

    return vectordb


vectordb = load_or_create_vectordb()


@app.route('/chat/', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    stream = data.get('stream', False)

    if stream:
        def generate():
            for chunk in generate_streaming_response(message, OPENAI_API_KEY, PROMPT_TEMPLATE, conversation_histories.get(message, [])):
                yield f"data: {chunk}\n\n"
        return Response(generate(), content_type='text/event-stream')

    search_results = vectordb.similarity_search(message, k=5)

    if search_results:
        pdf_extract = "\n".join([result.page_content for result in search_results])
        system_prompt = PROMPT_TEMPLATE + f"\nPDF Content:\n{pdf_extract}"
    else:
        system_prompt = PROMPT_TEMPLATE

    response = generate_response(message, OPENAI_API_KEY, system_prompt, conversation_histories.get(message, []))
    conversation_histories[message] = update_conversation_history(conversation_histories.get(message, []), message, response)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run()
