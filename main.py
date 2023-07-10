from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
import streamlit as st
import os
import tempfile

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

log = ""

st.subheader("Step 2 : Upload a PDF File")
uploaded_file = st.file_uploader("Select file", type=['pdf'])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    log = log + "\n Temporary Directory : " + temp_dir.name

    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    log = log + "\n Full File Path : " + temp_file_path

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
    log = log + "\n Create the temp file"

    # Create the PyPDFLoader using the temporary file path
    loader = PyPDFLoader(temp_file_path)
    log = log + "\n Load the Pdf file" 

    pages = loader.load_and_split()
    no_of_pages = len(pages)
    log = log + "\n" + uploaded_file.name + " was splitted into  : " + str(no_of_pages) + " page(s)" 
    
    if (no_of_pages>0):
        log = log + "\nFollowing are First Sample page content and ready to create embedding"
        log = log + "\n" + pages[0].page_content

        embeddings = OpenAIEmbeddings()
        log = log + "\nCreate LLM embedding"
        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                    persist_directory=".")
        log = log + "\nCreate Vector DB"
        vectordb.persist()
        pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                        vectordb, return_source_documents=False)
        log = log + "\nConnect LLM embedding and  Vector DB"
    

        with st.expander("For details, click here"):
            st.code(log)



        st.subheader("Step 3 : Enter the Prompt:")
        query = "What are the skillset?"
        query = st.text_area("Enter your prompt", query)
        if st.button("Generate"):
            with st.spinner("Generating ...."):
                result = pdf_qa({"question": query, "chat_history": ""})
                st.info(result)

        
