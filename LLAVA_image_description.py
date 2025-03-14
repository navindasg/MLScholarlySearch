import os
import json
from typing import Dict, Optional, List
from pathlib import Path
import base64
import requests
from PIL import Image

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image(
    image_directory: str = "./parser_output/extracted_images",
    output_file: Optional[str] = None,
    model_name: str = "llava:13b"
) -> Dict[str, str]:
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [
        f for f in Path(image_directory).iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"No images found in {image_directory}")
        return {}
    
    print(f"Found {len(image_files)} images to process")
    descriptions = {}
    
    try:
        test_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "Hello", "stream": False}
        )
        test_response.raise_for_status()
        print(f"Successfully connected to Ollama with model {model_name}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running and the model is available")
        return {}
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        try:
            base64_image = encode_image_base64(str(img_path))
            
            data = {
                "model": model_name,
                "prompt": "Please briefly describe this image. Focus on what it shows, including any visible text, diagrams, figures, or scientific visualizations. If this is from a scientific paper, explain what the image is depicting.",
                "stream": False,
                "images": [base64_image]
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "No description generated")
                descriptions[img_path.name] = description
                print(f"✓ Generated description for {img_path.name}")
            else:
                error_msg = f"Error from Ollama API: HTTP {response.status_code}, {response.text}"
                print(error_msg)
                descriptions[img_path.name] = f"Error: {error_msg}"
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            descriptions[img_path.name] = f"Error: {str(e)}"
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)
        print(f"Descriptions saved to {output_file}")
    
    return descriptions

if __name__ == "__main__":
    descriptions = describe_image(
        image_directory="./parser_output/extracted_images",
        output_file="./parser_output/image_descriptions.json",
        model_name="llava:13b"
    )
    
    print("\nSample descriptions:")
    for i, (img, desc) in enumerate(list(descriptions.items())[:3]):
        print(f"\n{img}:\n{desc[:150]}...")
