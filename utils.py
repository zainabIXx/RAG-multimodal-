import io
import os
import base64
import tempfile
import uuid
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import HumanMessage
from langchain.schema.document import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import InMemoryStore
from unstructured.documents.elements import Text, Table, CompositeElement
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize models
chat_model = ChatOpenAI(temperature=0, model="gpt-4o")
vectorstore = Chroma(collection_name="multimodal_rag", embedding_function=OpenAIEmbeddings())

# === PDF PROCESSING FUNCTIONS === #
def extract_pdf_elements(uploaded_file): # later change to filepath only
    """ Extract text, tables, and images from a PDF
    
    Parameters:
        uploaded_file (BytesIO): File uploaded via Streamlit
        
    Returns:
        text_data (List[str]): Text chunks from PDF
        table_data (List[str]): HTML table representations
        image_data (List[str]): Base64 encoded images
    """
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name  # Get the temporary file path
    
    # Partition PDF into chunks and extract images, tables, and text
    chunks = partition_pdf(filename=temp_file_path,
                           infer_table_structure=True,                   # Extract tables
                           strategy="hi_res",                            # Mandatory for table extraction
                           extract_image_block_types=["Image", "Table"], # Extract images and table images
                           extract_image_block_to_payload=True,          # True, extract base64 images for API
                           chunking_strategy="by_title",                 # Can also use 'basic'
                           max_characters=4000,                          # Defaults to 500
                           new_after_n_chars=3800, 
                           combine_text_under_n_chars=2000,              # Defaults to 0
                           )   

    # Create the empty list
    text_data, table_data, image_data = [], [], [] 
    
    # Extract Text
    text_data = [chunk.text for chunk in chunks if isinstance(chunk, Text)] 
    
    # Extract Tables (Including Nested in CompositeElement)
    for chunk in chunks:
        if isinstance(chunk, Table):
            table_data.append(chunk.metadata.text_as_html) # Extract table as HTML
        elif isinstance(chunk, CompositeElement):
            chunk_elements = chunk.metadata.orig_elements
            for el in chunk_elements:
                if "Table" in str(type(el)):
                    table_data.append(el.metadata.text_as_html)
             
    # Extract Images (Base64)
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    image_data.append(el.metadata.image_base64)
                    
    return text_data, table_data, image_data

    
# === SUMMARIZATION FUNCTIONS === #
def generate_text_summaries(texts, tables, summarize_texts=False):
    """ Generate summaries for tables and texts
    
    Parameters:
        texts (List[str]): Text chunks from PDF
        tables (List[str]): HTML table representations
        summarize_texts (bool): Boolean to summarize texts
    
    Returns:
        text_summaries (List[str]): Summaries for texts
        table_summaries (List[str]): Summaries of tables
    """
    
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    # Summarize chain
    summarize_chain = {"element": lambda x: x} | prompt | chat_model | StrOutputParser()
    
    # Initialize empty summaries
    text_summaries, table_summaries = [], []
    
    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    elif texts:
        text_summaries = texts # Return raw texts if summarization is disabled
    
    # Apply to table if tables are provided 
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 3})
    
    return text_summaries, table_summaries

def generate_image_summary(img_b64):
    """ Generate summaries for a list of base64-encoded images
    
    Parameters:
        img_b64 (List[str]): List of base64-encoded images
        
    Returns:
        image_summaries (List[str]): Summaries for each image
    """
    # Prompt 
    prompt_template = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""
    
    # Initialize empty summary
    image_summaries = []

    for img in img_b64:
        # Construct dynamic messages for each image
        messages = [
            ("user", [
                {"type": "text", "text": prompt_template},
                {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                },
            ])
        ]
        
        # Create ChatPromptTemplate dynamically
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | chat_model | StrOutputParser()
        
        # Generate summary 
        summaries = chain.invoke({"image": img})
        image_summaries.append(summaries)
    
    return image_summaries


# === MULTIVECTOR RETRIEVER === #
def create_multivector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """ Create a multimodal retriever using the provided vectorstore
    
    Parameters:
        vectorstore (VectorStore): A LangChain vectorstore for storing and retrieving embeddings
        text_summaries (List[str]): Summaries for texts
        texts (List[str]): Text chunks from PDF
        table_summaries (List[str]): Summaries of tables
        tables (List[str]): HTML table representations
        image_summaries (List[str]): Summaries for each image
        images (List[str]): Base64-encoded images
        
    Returns:
        multimodal_retriever (MultiModalRetriever): A multimodal retriever using the provided vectorstore
    """
    # Initalize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"
    
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
             
    # Add texts, tables, and images to the vectorstore and docstore
    if text_summaries: add_documents(retriever, text_summaries, texts)
    if table_summaries: add_documents(retriever, table_summaries, tables)
    if image_summaries: add_documents(retriever, image_summaries, images)
        
    return retriever

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64, text = [], []
    
    for doc in docs:
        try:
            # Validate if `doc` is a base64 image (raises an error if invalid)
            base64.b64decode(doc, validate=True)  
            b64.append(doc)
        except Exception:
            text.append(doc)  # If it's not a base64 image, it's text

    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    """ Build a prompt with text and base64-encoded image context.

    Parameters:
        kwargs (dict): Dictionary containing "context" and "question".

    Returns:
        ChatPromptTemplate: A structured prompt template.
    """
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # Merge text elements into a single string
    context_text = "\n".join(docs_by_type["texts"]) if docs_by_type["texts"] else ""

    # Construct prompt with context (text, tables, images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.

    Context: {context_text}
    Question: {user_question}
    """

    # Initialize prompt content
    prompt_content = [{"type": "text", "text": prompt_template}]

    # Append images if present
    for image in docs_by_type["images"]:
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )

    # Return structured prompt
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


# === MULTIMODAL RAG CHAIN === #
def query_multimodal_rag(retriever, query):
    """ Executes the Multimodal RAG pipeline for a user query
    
    Parameters:
        retriever (MultiVectorRetriever): The initialized retriever object
        query (str): The user's query
        
    Returns:
        dict: Retrieved documents & LLM-generated response
    """
    # Define LangChain processing chain
    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | chat_model
            | StrOutputParser()
        )
    )
    
    # Run the query through the chain
    response = chain_with_sources.invoke({"question": query})
    
    return response


# === IMAGE UTILITY FUNCTIONS === #
def decode_base64_image(img_base64):
    """Decode a base64-encoded image and return a PIL Image
    
    Parameters:
        img_base64 (str): Base64-encoded image string
        
    Returns:
        PIL.Image: Decoded image object
    """
    img_data = base64.b64decode(img_base64)
    return Image.open(io.BytesIO(img_data))

def resize_base64_image(base64_string, size=(128, 128)):
    """Resize an image encoded as Base64 string
    
    Parameters:
        base64_string (str): Base64-encoded image string
        size (tuple): Desired image dimensions (width, height)
        
    Returns:
        str: Resized image as a Base64-encoded string
    """
    img = decode_base64_image(base64_string)
    resized_img = img.resize(size, Image.LANCZOS)
    
    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    
    return base64.b64encode(buffered.getvalue()).decode("utf-8")