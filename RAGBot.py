import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import streamlit as st
import faiss
import os
from transformers import AutoTokenizer, AutoModel
import time

#Models
tokenizer_RAG = AutoTokenizer.from_pretrained("bert-base-uncased")
model_RAG = AutoModel.from_pretrained("bert-base-uncased")
model_QA = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer_QA = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#A function that fetches the context according to the query
def text_fetch(query_text, threshold=120):
    query_tokens = tokenizer_RAG(query_text, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model_RAG(**query_tokens).last_hidden_state.mean(dim=1).numpy()

    num_results = 1
    distances, indices = faiss_index.search(query_embedding, num_results)

    print(distances)

    # Check if distance is below the threshold
    if all(dist < threshold for dist in distances[0]):
        similar_texts = [texts_to_store[i] for i in indices[0]]
        return similar_texts[0]
    else:
        return "Unable to find the answer to your question."


#A function that comes up with an answer to the query with the help of a context
def question_answer(question, text):
    input_ids = tokenizer_QA.encode(question, text) #tokenize question and text as a pair
    tokens = tokenizer_QA.convert_ids_to_tokens(input_ids)   #string version of tokenized ids

    sep_idx = input_ids.index(tokenizer_QA.sep_token_id)  #segment IDs, #first occurence of [SEP] token

    num_seg_a = sep_idx+1 #number of tokens in segment A (question)
    num_seg_b = len(input_ids) - num_seg_a #number of tokens in segment B (text)

    segment_ids = [0]*num_seg_a + [1]*num_seg_b  #list of 0s and 1s for segment embeddings
    assert len(segment_ids) == len(input_ids)

    output = model_QA(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])) #model output using input_ids and segment_ids
    answer = ""
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    if answer.startswith("[SEP]" or "[CLS]"):
        answer = "Unable to find the answer to your question."
    return answer

#Streamlit 
image = 'icon.png'
user = 'user.png'
bot = 'Bot.jpg'

st.set_page_config(page_title='Plaksha Bot', page_icon = image, layout = 'centered', initial_sidebar_state = 'auto')

st.title("Plaksha Bot")

folder_path = "C:\\Users\\ankur\\Desktop\\Plaksha TLF\\Term 4\\Python for Data Science\\RAGBot"
texts_to_store = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            texts_to_store.append(text)

#Encode texts to be stored in the Faiss index
encoded_vectors = []
for text in texts_to_store:
    tokens = tokenizer_RAG(text, return_tensors="pt")
    with torch.no_grad():
        embedding = model_RAG(**tokens).last_hidden_state.mean(dim=1).numpy()
    encoded_vectors.append(embedding)
encoded_vectors = np.concatenate(encoded_vectors, axis=0)

#Initialize Faiss index
index_dimension = encoded_vectors.shape[1]  # Dimension of the encoded vectors
faiss_index = faiss.IndexFlatL2(index_dimension)  # L2 distance metric

#Add encoded vectors to Faiss index
faiss_index.add(encoded_vectors)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    avatar = bot if role == "assistant" else user  # Change "user" to the actual user avatar

    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt}) # Add user message to chat history
    with st.chat_message("user", avatar = user):
        st.markdown(prompt) # Display user message in chat message container

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar = bot):
        message_placeholder = st.empty()
        full_response = ""
        context = text_fetch(prompt)
        print(context) #To see which context it takes
        if context == "Unable to find the answer to your question.":
            assistant_response = "Unable to find the answer to your question."
        else:
            assistant_response = question_answer(prompt, context)
        
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌") # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response}) # Add assistant response to chat history
