from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
import streamlit as st
import os


# requirements.txt
# chromadb
# langchain
# pypdf
# streamlit


st.set_page_config(
    page_title="CSV Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: Step 1: Enter your OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key


st.caption("Import a PDF File")
uploaded_file = st.file_uploader("Step 2 : 📂 upload PDF file", type=['pdf'])
if uploaded_file is not None:
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load_and_split()
    no_of_pages = len(pages)
    st.caption(uploaded_file.name + " was splitted into  : " + str(no_of_pages) + " page(s)" )
    
    if (no_of_pages>0):
        st.caption("Following are First Sample page content and ready to create embedding") 
        st.info(pages[0].page_content)

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                    persist_directory=".")
        vectordb.persist()
        pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                        vectordb, return_source_documents=False)

    
        st.caption("Enter the Prompt:")
        query = "What are the skillset?"
        query = st.text_area("Enter your prompt", query)
        if st.button("Generate"):
            with st.spinner("Generating ...."):
                result = pdf_qa({"question": query, "chat_history": ""})
                st.info(result)

        