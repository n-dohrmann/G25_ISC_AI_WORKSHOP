__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Block 2: Create the Streamlit Application File (app.py)
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
from urllib.error import URLError
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader


DATA_URL = "https://huggingface.co/datasets/ISIntersystems/HealthcareData/resolve/main/HealthcareData.csv"


# --- Page Configuration ---
st.set_page_config(
    page_title="Clinical Note Category Finder",
    page_icon="#",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("# Note Category Finder")
st.markdown("""
Welcome! Please enter your clinical note, and our application will classify its category.
""")

# --- API Key Handling ---
openai_api_key = None

# Attempt to get API key from st.secrets (for deployed apps)
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if openai_api_key:
        st.sidebar.success("API key loaded from st.secrets!")
    else: # Handle case where secret exists but is empty
        st.sidebar.warning("OpenAI API Key found in st.secrets but it's empty. Please provide a valid key.")
except (KeyError, FileNotFoundError): # FileNotFoundError for local st.secrets.toml if used
    st.sidebar.info("OpenAI API Key not found in st.secrets. Please enter it below for this session.")

# Fallback to user input if not found in secrets or if secret was empty
if not openai_api_key:
    openai_api_key_input = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        key="api_key_input_sidebar",
        help="Your API key is used only for this session and not stored."
    )
    if openai_api_key_input:
        openai_api_key = openai_api_key_input

if not openai_api_key:
    st.warning("Please provide your OpenAI API Key in the sidebar to use the chat.")
    st.stop() # Stop execution if no API key is available


## Load in CSV data from hugging face (G25 specific)
def read_huggingface_csv(url: str) -> pd.DataFrame | None:
    """
    Reads a CSV file from a Hugging Face dataset URL into a pandas DataFrame.
    To get the correct URL:
    1. Navigate to the dataset repository on Hugging Face.
    2. Find the CSV file in the "Files and versions" tab.
    3. Click on the file.
    4. Right-click the "download" button and copy the link address.
    Args:
        url: The raw URL of the CSV file on Hugging Face.
    Returns:
        A pandas DataFrame containing the data from the CSV file,
        or None if an error occurs.
    """
    if not isinstance(url, str) or not url.startswith('http'):
        print("Error: Invalid URL provided. Please provide a valid HTTP/HTTPS URL.")
        return None

    try:
        print(f"Attempting to read CSV from: {url}")
        # pandas.read_csv can directly handle URLs
        df = pd.read_csv(url)
        print("Successfully read CSV into DataFrame.")
        return df
    except URLError as e:
        print(f"Error fetching the URL: {e}")
        print("Please check if the URL is correct and accessible.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        print("The file at the URL may not be a valid CSV.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def init_healthcare_df() -> pd.DataFrame:
    """
    initialize the healthcare data frame from hugging face 
    """
    return read_huggingface_csv(DATA_URL)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_fixed_chunk_docs():
    """
    // get return type here...?
    """
    df = init_healthcare_df()
    loader = DataFrameLoader(df, page_content_column="CLINICAL_NOTES_CLEANED")
    return loader.load()

def create_and_store_embeddings(documents_to_embed: list) -> Chroma:
    """
    Initializes an embedding model and creates a vector store from documents.

    This function takes a list of prepared text documents and converts them into
    numerical vectors (embeddings) using an AI model. It then stores these
    embeddings in a searchable Chroma vector database.

    Args:
        documents_to_embed: A list of chunked LangChain Document objects.
        api_key: The API key required to use the OpenAI embedding service.

    Returns:
        A Chroma vector store object that contains the documents and their embeddings.
    """

    # 1. Initialize the Embedding Model
    # An "embedding model" is a specialized AI that reads text and converts it into
    # a fixed-size list of numbers called a "vector". This vector captures the
    # semantic meaning of the text, allowing us to compare texts mathematically.
    # Here, we're using OpenAI's "text-embedding-3-small" model, which offers a great
    # balance of performance and cost.
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

    # 2. Create the Vector Store
    # A "vector store" or "vector database" is a special kind of database designed to
    # efficiently store and search for vectors.
    # The `Chroma.from_documents` function is a powerful helper that does two things at once:
    #   a. It uses the `embeddings_model` to create a vector for every document you provide.
    #   b. It stores both the original document content and its new vector in the Chroma database.
    # This process of creating vectors and storing them is often called "ingestion".
    # --- NOTE: Converting the documents into vectors takes time, for this example, expect about 30 seconds. ---
    vectorstore = Chroma.from_documents(
            documents = documents_to_embed,  # The list of prepared Document objects.
            embedding = embeddings_model,    # The model to use for the embedding process.
            # persist_directory="./chroma_db"  # Optional: You can uncomment this line to save the
                                               # database to a folder on your computer. This allows
                                               # you to load it again later without re-running the process if it is time consuming
        )

    # Print a confirmation message to the user so they know this long step is complete.
    print("\nEmbedding and ingestion complete.")

    # Return the fully populated vector store object, which is now ready to be used for searches.
    return vectorstore

def create_vector_db():
    """
    returns vector db to use in generation of rag response
    """
    fixed_chunk_docs = get_fixed_chunk_docs()
    return create_and_store_embeddings(fixed_chunk_docs)



def generate_rag_response(input_text: str, _vectorstore) -> str:
    """
    Generates a response using the RAG pattern.

    Args:
        input_text: The user's question.
        _vectorstore: The Chroma vector store containing the document embeddings.
        _openai_api_key: The API key for the OpenAI service.

    Returns:
        The AI-generated response as a string.
    """
    # A quick safety check to ensure the vector store has been created before we try to use it.
    if _vectorstore is None:
        return "Error: Vector store not available."

    # Initialize the Language Model (LLM) we'll use to generate the final answer.
    # 'temperature=0.7' controls the creativity of the model; higher values mean more creative,
    # lower values mean more deterministic and factual.
    llm = ChatOpenAI(temperature=0.7, api_key=openai_api_key)

    # Create a "retriever" from our vector store. A retriever is an object that can
    # "retrieve" documents from a database based on a query.
    # 'search_kwargs={"k": 3}' tells the retriever to find the top 3 most relevant
    # document chunks for any given question.
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5}) #k!!!! 3

    # Analyze the context below to answer the user's question.
    # Find the clinical note type for {question}


    # Define the prompt template. This is the heart of our instruction to the AI.
    # It sets the persona (a literary assistant), gives clear instructions on how to behave,
    # and defines where the retrieved 'context' and user's 'question' will be inserted.
    prompt_template = """You are a helpful assistant.
    Analyze the context below and determine the clinical note type for the user's note: {question}
    Explain the thought process

    The list of available clinical note types are:
      Well child visit (procedure)
      Encounter for check up (procedure)
      Encounter for symptom (procedure)
      Emergency room admission (procedure)
      Non-urgent orthopedic admission (procedure)
      Hospital admission (procedure)
      Administration of vaccine to produce active immunity (procedure)
      Encounter for problem (procedure)
      General examination of patient (procedure)
      Follow-up encounter (procedure)
      Urgent care clinic (environment)
      Prenatal visit (regime/therapy)
      Consultation for treatment (procedure)
      Prenatal initial visit (regime/therapy)
      Obstetric emergency hospital admission (procedure)
      Postnatal visit (regime/therapy)
      Patient-initiated encounter (procedure)
      Patient encounter procedure (procedure)
      Admission to intensive care unit (procedure)
      Death Certification
      Emergency treatment (procedure)
      Emergency hospital admission (procedure)
      Admission to surgical department (procedure)
      Telemedicine consultation with patient (procedure)
      Admission to hospice (procedure)
      Drug rehabilitation and detoxification (regime/therapy)
      Office Visit
      Domiciliary or rest home patient evaluation and management (procedure)
      Hospital admission for isolation (procedure)
      Admission to surgical transplant department (procedure)
      Postoperative follow-up visit (procedure)
      Home visit (procedure)

    Context:
    {context}


    Answer:"""

    # Create a ChatPromptTemplate object from the string template defined above.
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Now, we create the full RAG chain using LangChain Expression Language (LCEL).
    # The '|' (pipe) operator connects the different components in a sequence.
    rag_chain = (
        # This first step runs in parallel. It creates a dictionary containing the 'context' and 'question'.
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # 1a. "context": The user's input goes to the 'retriever', which finds relevant docs.
        #     The list of docs is then passed to our 'format_docs' function to create the context string.
        # 1b. "question": 'RunnablePassthrough' simply takes the user's original input (the question)
        #     and passes it through unchanged.

        | prompt          # 2. The dictionary from the previous step is "piped" into the 'prompt' template.
                          #    This fills in the {context} and {question} placeholders.

        | llm             # 3. The fully-formatted prompt is sent to the language model ('llm') to generate an answer.

        | StrOutputParser() # 4. The model's output (a chat message object) is converted into a simple string.
    )

    # Use a try-except block for robust error handling, especially for API calls.
    try:
        # ".invoke()" is the command that runs the entire chain with the user's question.
        response = rag_chain.invoke(input_text)
        return response
    except Exception as e:
        # If anything goes wrong during the chain execution, print the error.
        print(f"Error generating response: {e}")
        return "Sorry, an error occurred while generating the response."

# put this somewhere better lol??
VECTOR_DB = create_vector_db()


# --- LangChain Setup (Cached for efficiency) ---
@st.cache_resource # Caches the LLM and prompt template
def get_langchain_components(_api_key_for_cache): # Parameter ensures cache reacts to API key changes if necessary
    """Initializes and returns the LangChain LLM and prompt template."""
    llm = ChatOpenAI(openai_api_key=_api_key_for_cache, model_name="gpt-4o-mini")
    
    prompt_template_str = """
    You are a knowledgeable and friendly AI assistant.
    Your goal is to provide clear, concise, and helpful answers to the user's questions.
    If you don't know the answer to a specific question, it's better to say so rather than inventing one.
    User Question: {user_input}
    AI Response:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    return llm, prompt

#try:
#    llm, prompt_template = get_langchain_components(openai_api_key)
#except Exception as e:
#    st.error(f"Failed to initialize AI components. Error: {e}. Check your API key and model access.")
#    st.stop()

# --- Initialize session state for storing chat messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display existing chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and AI Response Logic ---
if user_query := st.chat_input("Enter the clinical notes for the patient visit"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # For "Thinking..." message and then the actual response
        with st.spinner("AI is thinking..."):
            try:

                response = generate_rag_response(user_query, VECTOR_DB)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                #chain = prompt_template | llm
                #ai_response_message = chain.invoke({"user_input": user_query})
                #ai_response_content = ai_response_message.content
                
                #message_placeholder.markdown(ai_response_content)
                # Add AI response to session state
                # st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Sidebar Options ---
with st.sidebar:
    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun() # Rerun to update the UI immediately

    st.markdown("---")
    st.subheader("About")
    st.info(
        "The **Note Category Finder** takes in a clinical note,"
        "and uses a RAG AI system to determine an appropriate category."
    )
