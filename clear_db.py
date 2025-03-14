from langchain_community.vectorstores import Chroma

vectorstore = Chroma(persist_directory="./vector_db")

#delete all data
vectorstore.delete()
