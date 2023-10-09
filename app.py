import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import openai

# from ludwig.api import LudwigModel   # XXX
import transformers
from langchain.llms import HuggingFacePipeline
import torch

import sys   #x

MODELS_EMB = ("Local Instructor", "OpenAI Ada")
MODELS_CON = ("Local Llama-7b", "OpenAI GPT-4", "HF FLAN-T5-XXL")

def summarize_text(vectordb):
    print( f'  ### method: {sys._getframe().f_code.co_name}' )

    llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', model_kwargs={'deployment_id': 'gpt-35-turbo'})

#    MODEL = "tiiuae/falcon-7b-instruct"
#    model_temperature=0.1  # to do
#    model_temperature = max( 1e-8, model_temperature )
#    llm = HuggingFaceHub(repo_id=MODEL, model_kwargs={"temperature": model_temperature, "max_length":2048})

    chain = load_summarize_chain(llm, chain_type="stuff")
    search = vectordb.similarity_search("Summary of the file")
    print(f'LLM: {llm}')
    print(f'len(search)={len(search)}, search[0]: {search[0].page_content}')
    summary = chain.run(input_documents=search, question="Write a summary within 200 words.")
    print( f'Summary: #chars = {len(summary)}  #words = {len(summary.split(" "))}' )
    return summary

def get_pdf_text(pdf_docs):
#    print( f'  ### method: {sys._getframe().f_code.co_name} \t pdf_docs = {pdf_docs}' )
    print( f'  ### method: {sys._getframe().f_code.co_name} \t pdf_docs[].name = {[pdf_doc.name for pdf_doc in pdf_docs]}' )

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    print( f'  ### method: {sys._getframe().f_code.co_name} \t len(text) = {len(text)}' )

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
#        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# "Local Instructor", "OpenAI Ada"
def get_vectorstore(text_chunks, embedding_model):
    print( f'  ### method: {sys._getframe().f_code.co_name} \t len(text_chunks) = {len(text_chunks)}  \t embedding_model = {embedding_model}' )

    if embedding_model == "OpenAI Ada":
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    elif embedding_model == "Local Instructor":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    else:
        raise ValueError(f"Invalid language model choice. Supported options: {MODELS_EMB}.")

    print(f'embeddings: {embeddings}')

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# "Local Llama-7b", "OpenAI GPT-4", "HF FLAN-T5-XXL"
def get_conversation_chain(vectorstore, conversational_model, model_temperature=0.5):
    print( f'  ### method: {sys._getframe().f_code.co_name} \t conversational_model = {conversational_model}' )
    if conversational_model == "OpenAI GPT-4":
        llm = ChatOpenAI(temperature=model_temperature, model_name="gpt-4", deployment_id="gpt-4")
    elif conversational_model == "HF FLAN-T5-XXL":
#        MODEL = "meta-llama/Llama-2-7b-hf"
#        MODEL = "tiiuae/falcon-7b-instruct"    # always empty response?
        MODEL = "google/flan-t5-xxl"
        model_temperature = max( 1e-8, model_temperature )
        llm = HuggingFaceHub(repo_id=MODEL, model_kwargs={"temperature": model_temperature, "max_length":2048})
    elif conversational_model == "Local Llama-7b":

        MODEL = "meta-llama/Llama-2-7b-chat-hf"

        if model_temperature < 1e-6:
            pipeline = transformers.pipeline(
                task="text-generation", #task
                model=MODEL,
                # tokenizer=tokenizer,
                return_full_text=True,  # langchain expects the full text
                # stopping_criteria=stopping_criteria,  # without this model rambles during chat
                repetition_penalty=1.1,  # without this output begins repeating
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                max_length=4000,
                do_sample=False,
                #            top_k=10,
                num_return_sequences=1,
                # eos_token_id=tokenizer.eos_token_id
            )
        else:
            pipeline = transformers.pipeline(
                task="text-generation", #task
                model=MODEL,
                # tokenizer=tokenizer,
                return_full_text=True,  # langchain expects the full text
                # stopping_criteria=stopping_criteria,  # without this model rambles during chat
                temperature=model_temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                repetition_penalty=1.1,  # without this output begins repeating
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                max_length=4000,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                # eos_token_id=tokenizer.eos_token_id
            )
            
        print("Loading Llama2 ... begin")
        llm = HuggingFacePipeline(pipeline = pipeline, model_id=MODEL)
        print("Loading Llama2 ... end")
        
    print(f'LLM: -- {llm} --')
        
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
#    print("mem", memory.load_memory_variables({}))
    return conversation_chain


def handle_userinput(user_question):
    print( f'  ### method: {sys._getframe().f_code.co_name} \t user_question = {user_question}' )
#    print( 'UU', len(user_question), len(user_question.split(' ')) )
    response = st.session_state.conversation({'question': user_question})
    print( f'response: {response}' )
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
#        print('msg', message.content)
#        print('ut ', user_template.replace( "{{MSG}}", message.content) ) 
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    print( f'  ### method: {sys._getframe().f_code.co_name}' )
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":brain:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :brain:")
    user_question = st.text_input("Ask a question about your documents:")

    

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        # Choose the embedding model
        embedding_model = st.radio("Choose Embedding Model", MODELS_EMB)

        # Choose the conversational model
        conversational_model = st.radio("Choose Conversational Model", MODELS_CON)

        # Choose the model temperature
        model_temperature = st.slider("Choose Model Temperature", 0.0, 1.0, 0.0, 0.1)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, embedding_model)

#                # summarize the text
#                st.write("Summary of the file")
#                st.write(summarize_text(vectorstore))


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, conversational_model, model_temperature)
                
                


if __name__ == '__main__':
    main()
