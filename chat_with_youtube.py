# Import necessary libraries
import os
import tempfile
import whisper
from pytube import YouTube
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Set Pinecone API key
pinecone_api_key = "xxxxxxxxxxxxxxxxx"
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# openai api key
OPENAI_API_KEY = ""  # Replace with your OpenAI API key

# youtube url
YOUTUBE_URL = "" # Replace with the URL of the YouTube video

# Function to set up the OpenAI model
def setup_openai_model(api_key, model_name):
    try:
        model = ChatOpenAI(openai_api_key=api_key, model=model_name)
        parser = StrOutputParser()
        chain = model | parser
        return chain, parser
    except Exception as e:
        print(f"Error setting up OpenAI model: {e}")
        return None, None

# Function to define the prompt template
def create_prompt_template():
    try:
        template = """
        Answer the question based on the context below. If you can't
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt
    except Exception as e:
        print(f"Error creating prompt template: {e}")
        return None

# Function to download and transcribe a YouTube video
def download_and_transcribe_video(video_url, transcript_file):
    try:
        if not os.path.exists(transcript_file):
            youtube_video = YouTube(video_url)
            audio_stream = youtube_video.streams.filter(only_audio=True).first()
            whisper_model = whisper.load_model("base")

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = audio_stream.download(output_path=temp_dir)
                transcription = whisper_model.transcribe(file_path, fp16=False)["text"].strip()

                with open(transcript_file, "w") as file:
                    file.write(transcription)
    except Exception as e:
        print(f"Error downloading or transcribing video: {e}")

# Function to load and split the transcription into documents
def load_and_split_transcription(transcript_file, chunk_size, chunk_overlap):
    try:
        with open(transcript_file) as file:
            transcription = file.read()

        loader = TextLoader(transcript_file)
        text_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(text_documents)
        return documents
    except Exception as e:
        print(f"Error loading or splitting transcription: {e}")
        return []

# Function to set up the vector embeddings
def setup_embeddings(api_key):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return embeddings
    except Exception as e:
        print(f"Error setting up embeddings: {e}")
        return None

# Function to set up the Pinecone vector store
def setup_pinecone_store(documents, embeddings, index_name):
    try:
        pinecone = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
        return pinecone
    except Exception as e:
        print(f"Error setting up Pinecone vector store: {e}")
        return None

# Function to set up the retrieval-augmented chain
def setup_retrieval_chain(pinecone_store, prompt_template, openai_model, parser):
    try:
        chain = (
            {"context": pinecone_store.as_retriever(), "question": RunnablePassthrough()}
            | prompt_template
            | openai_model
            | parser
        )
        return chain
    except Exception as e:
        print(f"Error setting up retrieval chain: {e}")
        return None
# Main function
def main():
    # Set up OpenAI API key and model
    openai_api_key = OPENAI_API_KEY
    model_name = "gpt-3.5-turbo"
    openai_chain,parser = setup_openai_model(openai_api_key, model_name)

    # Set up prompt template
    prompt_template = create_prompt_template()

    # Download and transcribe the YouTube video
    video_url = YOUTUBE_URL
    transcript_file = "transcription1.txt"
    download_and_transcribe_video(video_url, transcript_file)

    # Load and split the transcription into documents
    chunk_size = 1000
    chunk_overlap = 20
    documents = load_and_split_transcription(transcript_file, chunk_size, chunk_overlap)

    # Set up vector embeddings
    embeddings = setup_embeddings(openai_api_key)

    # Set up Pinecone vector store
    index_name = "testindex"
    pinecone_store = setup_pinecone_store(documents, embeddings, index_name)

    # Set up retrieval-augmented chain
    retrieval_chain = setup_retrieval_chain(pinecone_store, prompt_template, openai_chain, parser)

    # Use the chain to answer a question
    question = "what sam atlman think about elon musk" # replace with your question
    result = retrieval_chain.invoke(question)
    print(result)

if __name__ == "__main__":
    main()