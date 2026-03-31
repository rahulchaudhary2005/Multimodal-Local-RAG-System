from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from multimodal import image_to_text
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load embedding
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectordb.as_retriever()

# Load LOCAL LLM (NO API)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt
prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question:
{question}
""")

# RAG chain (NEW STYLE)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# Multimodal Query
def query_system(user_input, image_path=None):
    if image_path:
        image_desc = image_to_text(image_path)
        user_input = f"{user_input} | Image: {image_desc}"

    return rag_chain.invoke(user_input)


# -------- RUN --------
if __name__ == "__main__":
    print("🔥 RAG Ready")

    while True:
        query = input("Query: ")
        img = input("Image path (optional): ")

        img = img if img.strip() != "" else None

        result = query_system(query, img)
        print("\n💡 Answer:", result)