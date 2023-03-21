import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.indexes import VectorstoreIndexCreator

load_dotenv(".env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    loader = TextLoader("./state_of_the_union.txt")
    llm = OpenAI(temperature=0.9)
    index = VectorstoreIndexCreator().from_loaders([loader])

    query = "What did the president say about Ketanji Brown Jackson"
    index.query(query)
