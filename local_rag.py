from langchain.retrievers import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import time
import json
import os
import re

class ThermodynamicProperty(BaseModel):
    molecule_name: str = Field(description="The name of the molecule or compound")
    melting_point: Optional[str] = Field(None, description="Melting point (Tm) of the molecule, including value and unit")
    crystallization_temperature: Optional[str] = Field(None, description="Crystallization temperature of the molecule, including value and unit")
    crystallization_driving_force: Optional[str] = Field(None, description="Crystallization driving force (ΔGc) of the molecule, including value and unit")
    source_context: str = Field(description="The relevant portion of text from which this information was extracted")

class ThermodynamicResults(BaseModel):
    properties: List[ThermodynamicProperty] = Field(description="List of thermodynamic properties extracted from the document")

def setup_model():
    llm = ChatOllama(
        model="deepseek-r1:32b", 
        temperature=0.1,
        top_p=0.95,
        repeat_penalty=1.15,
        max_tokens=1024,
        keep_alive="3h"
    )
    return llm

examples = [
    {
        "input": "Text discusses poly(ethylene glycol) (PEG) with a melting point of 65°C and shows crystallization at 40°C under a driving force (ΔGc) of -3.2 kJ/mol.",
        "output": """
{
  "properties": [
    {
      "molecule_name": "poly(ethylene glycol) (PEG)",
      "melting_point": "65°C",
      "crystallization_temperature": "40°C",
      "crystallization_driving_force": "-3.2 kJ/mol",
      "source_context": "Text discusses poly(ethylene glycol) (PEG) with a melting point of 65°C and shows crystallization at 40°C under a driving force (ΔGc) of -3.2 kJ/mol."
    }
  ]
}"""
    },
    {
        "input": "Polylactic acid (PLA) was studied extensively. Results indicated a Tm of 175°C. The crystallization occurred at 105°C with a thermodynamic driving force ΔGc of approximately -5.8 kJ/mol.",
        "output": """
{
  "properties": [
    {
      "molecule_name": "Polylactic acid (PLA)",
      "melting_point": "175°C",
      "crystallization_temperature": "105°C",
      "crystallization_driving_force": "-5.8 kJ/mol",
      "source_context": "Polylactic acid (PLA) was studied extensively. Results indicated a Tm of 175°C. The crystallization occurred at 105°C with a thermodynamic driving force ΔGc of approximately -5.8 kJ/mol."
    }
  ]
}"""
    }
]

def main():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    db = Chroma(
        persist_directory="./vector_db",
        embedding_function=embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    llm = setup_model()
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ]),
        examples=examples,
    )
    template = """You are a specialized research assistant tasked with extracting specific thermodynamic properties from scientific texts. 
    
    Identify any information about:
    1. Melting point (Tm)
    2. Crystallization temperature
    3. Crystallization driving force (ΔGc)
    
    For each property found, list the associated molecule or compound name. Only extract information that's explicitly mentioned in the text.
    
    IMPORTANT: Always include the units for temperature and driving force values. Pay special attention to whether temperatures are reported in Celsius (°C) or Kelvin (K). Always capture the units as they appear in the text.
    
    Here's the text to analyze:
    
    {context}
    
    Extract all thermodynamic properties mentioned above with their associated molecules. Format your response as a JSON object following this structure:
    
    ```json
    {{
      "properties": [
        {{
          "molecule_name": "name of molecule or compound",
          "melting_point": "value with unit (e.g., 350K or 65°C)",
          "crystallization_temperature": "value with unit (e.g., 300K or 40°C)",
          "crystallization_driving_force": "value with unit (e.g., -3.2 kJ/mol)",
          "source_context": "relevant portion of text from which this information was extracted"
        }},
        // Additional molecules as needed
      ]
    }}
    ```
    
    If no properties are found, return an empty properties list. Only include fields for which values are explicitly provided in the text. Do not hallucinate or infer values not explicitly stated.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        few_shot_prompt,
        ("human", "{context}")
    ])
    chain = (
        {"context": retriever}
        | prompt
        | llm
    )
    parser = PydanticOutputParser(pydantic_object=ThermodynamicResults)
    def standardize_temperature(temp_str):
        if not temp_str:
            return temp_str
        if '°C' in temp_str or 'C' in temp_str:
            return temp_str
        kelvin_pattern = r'(\d+\.?\d*)\s*(?:K|Kelvin)'
        match = re.search(kelvin_pattern, temp_str)
        
        if match:
            kelvin_value = float(match.group(1))
            celsius_value = round(kelvin_value - 273.15, 2)
            temp_celsius = f"{celsius_value}°C (converted from {kelvin_value}K)"
            return temp_celsius
        
        return temp_str
    def process_results(results):
        try:
            start_idx = results.find('{')
            end_idx = results.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = results[start_idx:end_idx]
                parsed_result = parser.parse(json_str)
                for prop in parsed_result.properties:
                    prop.melting_point = standardize_temperature(prop.melting_point)
                    prop.crystallization_temperature = standardize_temperature(prop.crystallization_temperature)
                
                return parsed_result
            else:
                return parser.parse('{"properties": []}')
        except Exception as e:
            print(f"Error parsing results: {e}")
            return ThermodynamicResults(properties=[])

    def search_thermodynamic_properties(query="thermodynamic properties melting point crystallization"):
        print(f"Searching for: {query}")
        print("Retrieving relevant documents...")
        
        start_time = time.time()
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        raw_results = llm.invoke(prompt.format(context=context))
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.2f} seconds")
        results = process_results(raw_results.content)
        with open("thermodynamic_properties.json", "w") as f:
            json_content = json.dumps(results.model_dump(), indent=2)
            f.write(json_content)
        print(f"Found {len(results.properties)} molecules with thermodynamic properties")
        print("Results saved to thermodynamic_properties.json")     
        return results

    search_terms = [
        "melting point Tm polymer crystallization",
        "crystallization temperature polymer thermodynamics",
        "crystallization driving force ΔGc polymer",
        "thermodynamic properties crystallization Tm ΔGc"
    ]
    
    all_results = ThermodynamicResults(properties=[])
    
    for term in search_terms:
        results = search_thermodynamic_properties(term)
        existing_molecules = {prop.molecule_name for prop in all_results.properties}
        for prop in results.properties:
            if prop.molecule_name not in existing_molecules:
                all_results.properties.append(prop)
                existing_molecules.add(prop.molecule_name)
    
    with open("all_thermodynamic_properties.json", "w") as f:
        json_content = json.dumps(all_results.model_dump(), indent=2)
        f.write(json_content)
    
    print(f"Total unique molecules found: {len(all_results.properties)}")
    print("All results saved to all_thermodynamic_properties.json")
    for i, prop in enumerate(all_results.properties):
        print(f"\nMolecule {i+1}: {prop.molecule_name}")
        if prop.melting_point:
            print(f"  Melting Point (Tm): {prop.melting_point}")
        if prop.crystallization_temperature:
            print(f"  Crystallization Temperature: {prop.crystallization_temperature}")
        if prop.crystallization_driving_force:
            print(f"  Crystallization Driving Force (ΔGc): {prop.crystallization_driving_force}")

if __name__ == "__main__":
    main()
