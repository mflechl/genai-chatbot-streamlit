[![CI](https://github.com/mflechl/genai-chatbot-streamlit/actions/workflows/main.yml/badge.svg)](https://github.com/mflechl/genai-chatbot-streamlit/actions/workflows/main.yml)

# genai-chatbot-streamlit
Chatbot using LangChain and LLMs+RAG with the option for the user to upload and talk to their PDFs

# to set up
```
git clone git@github.com:mflechl/genai-chatbot-streamlit.git

python3 -m venv env
source env/bin/activate

pip install --upgrade pip &&\
  pip install -r requirements.txt
```

# to run
```
streamlit run app.py  [--server.port NNNN]
```
