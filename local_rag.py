from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import List, Dict, Optional
import json
import os
from table_handler import TableHandler

def setup_model(model_name: str = "gemma3:27b") -> ChatOllama:
    """Setup the LLM model with specific parameters."""
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        format="json",
        system="""You are a scientific assistant specialized in extracting thermodynamic properties from materials science texts.
        Your task is to output valid JSON containing thermodynamic properties for ALL molecules mentioned in the text.
        You MUST NOT include any text before or after the JSON.
        For each molecule found, extract:
        - melting_point (in °C or K)
        - crystallization_temperature (in °C or K)
        - crystallization_driving_force (in kJ/mol)
        If a property is not found for a molecule, use null for that value.
        Return an array of objects, where each object represents one molecule and its properties."""
    )

def setup_embeddings() -> OllamaEmbeddings:
    """Setup the embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def setup_vector_store(embeddings: OllamaEmbeddings) -> Chroma:
    """Setup the vector store."""
    return Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings
    )

def setup_retriever(vector_store: Chroma) -> Chroma:
    """Setup the retriever with basic similarity search."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5  # Retrieve more documents
        }
    )

# Example few-shot prompts
examples = [
    {
        "input": "What are the thermodynamic properties of polyethylene and polypropylene?",
        "output": """[
            {
                "molecule": "polyethylene",
                "melting_point": "130°C",
                "crystallization_temperature": "110°C",
                "crystallization_driving_force": "-2.0 kJ/mol"
            },
            {
                "molecule": "polypropylene",
                "melting_point": "165°C",
                "crystallization_temperature": "100°C",
                "crystallization_driving_force": "-2.5 kJ/mol"
            }
        ]"""
    },
    {
        "input": "Find all materials and their crystallization properties.",
        "output": """[
            {
                "molecule": "polyethylene",
                "melting_point": null,
                "crystallization_temperature": "110°C",
                "crystallization_driving_force": "-2.0 kJ/mol"
            },
            {
                "molecule": "polypropylene",
                "melting_point": null,
                "crystallization_temperature": "100°C",
                "crystallization_driving_force": "-2.5 kJ/mol"
            }
        ]"""
    }
]

def create_prompt() -> ChatPromptTemplate:
    """Create the prompt template with few-shot examples."""
    # Create the few-shot prompt template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ]),
        examples=[
            {
                "input": "What are the thermodynamic properties of polyethylene and polypropylene?",
                "output": """[
                    {
                        "molecule": "polyethylene",
                        "melting_point": "130°C",
                        "crystallization_temperature": "110°C",
                        "crystallization_driving_force": "-2.0 kJ/mol"
                    },
                    {
                        "molecule": "polypropylene",
                        "melting_point": "165°C",
                        "crystallization_temperature": "100°C",
                        "crystallization_driving_force": "-2.5 kJ/mol"
                    }
                ]"""
            },
            {
                "input": "Find all materials and their crystallization properties.",
                "output": """[
                    {
                        "molecule": "polyethylene",
                        "melting_point": null,
                        "crystallization_temperature": "110°C",
                        "crystallization_driving_force": "-2.0 kJ/mol"
                    },
                    {
                        "molecule": "polypropylene",
                        "melting_point": null,
                        "crystallization_temperature": "100°C",
                        "crystallization_driving_force": "-2.5 kJ/mol"
                    }
                ]"""
            }
        ],
    )

    # Create the main prompt template
    template = """You are a scientific assistant specialized in extracting thermodynamic properties from materials science texts.
    Your task is to output valid JSON containing thermodynamic properties for ALL molecules mentioned in the text.
    
    For each molecule found, extract these specific properties:
    - melting_point: Look for values with units °C or K
    - crystallization_temperature: Look for values with units °C or K
    - crystallization_driving_force: Look for values with units kJ/mol
    
    If a property is not found for a molecule, use null for that value.
    Return an array of objects, where each object represents one molecule and its properties.
    Always include the units in the values.

    Context:
    {context}

    Question: {question}

    Answer:"""

    # Combine the templates
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        few_shot_prompt,
        ("human", "{question}")
    ])

    return prompt

def query_documents(query: str, k: int = 5) -> str:
    """Query the documents and return a response."""
    try:
        # Get relevant documents
        docs = retriever.invoke(query)
        
        # Get relevant tables
        tables = table_handler.list_tables()
        relevant_tables = []
        for table in tables:
            if any(term in table.lower() for term in ['temperature', 'melting', 'crystallization', 'force']):
                table_data = table_handler.query_table(table)
                if not table_data.empty:
                    relevant_tables.append(f"Table {table}:\n{table_data.to_string()}\n")
        
        # Get relevant image descriptions
        image_tables = [t for t in tables if t.startswith('image_')]
        relevant_images = []
        for img_table in image_tables:
            img_data = table_handler.query_table(img_table)
            if not img_data.empty and any(term in img_data['description'].iloc[0].lower() 
                                        for term in ['temperature', 'melting', 'crystallization', 'force', 'graph', 'plot']):
                relevant_images.append(f"Image {img_table}:\n{img_data['description'].iloc[0]}\n")
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Add relevant tables and images to context
        if relevant_tables:
            context += "\n\nRelevant Tables:\n" + "\n".join(relevant_tables)
        if relevant_images:
            context += "\n\nRelevant Images:\n" + "\n".join(relevant_images)
        
        # Create prompt with context
        prompt = f"""Based on the following context, extract thermodynamic properties in JSON format. 
        Focus on finding:
        1. Melting points
        2. Crystallization temperatures
        3. Crystallization driving forces
        
        Context:
        {context}
        
        Respond with a JSON object containing arrays of objects with these properties:
        {{
            "melting_points": [
                {{
                    "material": "name of material",
                    "value": "temperature value",
                    "unit": "°C or K"
                }}
            ],
            "crystallization_temperatures": [
                {{
                    "material": "name of material",
                    "value": "temperature value",
                    "unit": "°C or K"
                }}
            ],
            "crystallization_driving_forces": [
                {{
                    "material": "name of material",
                    "value": "force value",
                    "unit": "kJ/mol"
                }}
            ]
        }}
        
        If no properties are found, return an empty array for that category.
        """
        
        # Get response from model
        response = model.invoke(prompt)
        
        # Extract content from AIMessage
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Ensure response is valid JSON
        try:
            json_response = json.loads(response_text)
            return json.dumps(json_response, indent=2)
        except json.JSONDecodeError:
            return json.dumps({
                "melting_points": [],
                "crystallization_temperatures": [],
                "crystallization_driving_forces": []
            }, indent=2)
            
    except Exception as e:
        print(f"Error in query_documents: {e}")
        return json.dumps({
            "error": str(e),
            "melting_points": [],
            "crystallization_temperatures": [],
            "crystallization_driving_forces": []
        }, indent=2)

def main():
    """Main function to test the RAG system."""
    # Initialize components
    embeddings = setup_embeddings()
    vector_store = setup_vector_store(embeddings)
    global retriever, model, table_handler
    retriever = setup_retriever(vector_store)
    model = setup_model()
    table_handler = TableHandler()
    
    try:
        # Example usage
        question = "Find all molecules and their thermodynamic properties (melting point, crystallization temperature, and crystallization driving force)."
        answer = query_documents(question)
        print("\nFinal Answer:", answer)
    finally:
        # Clean up resources
        table_handler.close()

if __name__ == "__main__":
    main()
