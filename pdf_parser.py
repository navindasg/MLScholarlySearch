import os
import fitz
from PIL import Image
import io
import numpy as np
import cv2
import statistics
import json
import shutil
import csv
import pdfplumber
import pandas as pd
from typing import List, Dict, Optional
from table_handler import TableHandler

def extract_pdf_data(pdf_path, 
                     output_file=None, 
                     output_images_folder='./parser_output/extracted_images', 
                     output_tables_folder='./parser_output/extracted_tables',
                     image_area_threshold=1000,
                     color_similarity_threshold=0.9):
    
    if output_file is None:
        pdf_filename = os.path.basename(pdf_path)
        if pdf_filename.lower().endswith('.pdf'):
            pdf_filename = pdf_filename[:-4]
        output_file = os.path.join('./parser_output', f"{pdf_filename}.txt")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    """
    Extracts text, images, and tables from a PDF file and saves everything to one output file.
    
    Parameters:
      pdf_path (str): Path to the PDF file.
      output_file (str): Path to the single output text file.
      output_images_folder (str): Directory where extracted images will be temporarily saved.
      output_tables_folder (str): Directory where extracted tables will be temporarily saved.
      image_area_threshold (int): Minimum area (width x height in pixels) required to save an image.
      color_similarity_threshold (float): Maximum percentage of pixels allowed to be similar to the most dominant color.
    """
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_tables_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    file_exists = os.path.exists(output_file)
    
    with open(output_file, "a", encoding="utf-8") as txt_file:
        if file_exists:
            txt_file.write("\n\n" + "="*50 + "\n")
            txt_file.write(f"NEW EXTRACTION: {os.path.basename(pdf_path)}\n")
            txt_file.write("="*50 + "\n\n")
        
        txt_file.write("=== PDF TEXT CONTENT ===\n\n")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            txt_file.write(f"--- Page {page_num + 1} ---\n")
            txt_file.write(text)
            txt_file.write("\n\n")
    print("Text extraction complete. Saved to", output_file)
    
    # Image Extraction
    image_counter = 1
    extracted_image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width * height < image_area_threshold:
                continue
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            img_np = np.array(pil_image)
            if img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            hist, _ = np.histogram(magnitude, bins=50)
            gradient_smoothness = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
            line_count = 0 if lines is None else len(lines)
            resized = cv2.resize(img_cv, (100, 100))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h_bins = cv2.calcHist([hsv], [0], None, [18], [0, 180])
            s_bins = cv2.calcHist([hsv], [1], None, [10], [0, 256])
            v_bins = cv2.calcHist([hsv], [2], None, [10], [0, 256])
            unique_h = np.sum(h_bins > 5)
            unique_s = np.sum(s_bins > 5)
            unique_v = np.sum(v_bins > 5)
            color_variety = unique_h + unique_s + unique_v
            print(f"Image analysis for page {page_num + 1}:")
            print(f"  Edge density: {edge_density:.4f}, Line count: {line_count}")
            print(f"  Color variety: {color_variety}, Gradient smoothness: {gradient_smoothness:.4f}")
            is_solid_color = edge_density < 0.05 and color_variety < 8
            is_gradient = edge_density < 0.1 and gradient_smoothness < 0.8 and line_count < 5
            is_graph = (edge_density > 0.05 or line_count > 10) and color_variety > 5
            if is_solid_color:
                print(f"Skipping solid color image on page {page_num + 1}")
                continue
            elif is_gradient and not is_graph:
                print(f"Skipping gradient image on page {page_num + 1}")
                continue
            elif not is_graph and edge_density < 0.08 and line_count < 8:
                print(f"Skipping non-graph image on page {page_num + 1}")
                continue
            image_ext = base_image["ext"]
            image_filename = os.path.join(output_images_folder, f"image_{page_num + 1}_{image_counter}.{image_ext}")
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
            extracted_image_paths.append(image_filename)
            print(f"Extracted image saved as {image_filename}")
            image_counter += 1
    print("Image extraction complete. Saved in folder", output_images_folder)
    
    # Table Extraction using pdfplumber
    extracted_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        # Convert table to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_csv_path = os.path.join(output_tables_folder, f"table_{page_num}_{table_num}.csv")
                        df.to_csv(table_csv_path, index=False)
                        extracted_tables.append(table_csv_path)
                        print(f"Extracted table saved as {table_csv_path}")
        print("Table extraction complete. Saved in folder", output_tables_folder)
    except Exception as e:
        print("Error extracting tables:", e)
    
    if extracted_image_paths:
        from LLAVA_image_description import describe_image
        temp_json_path = "./parser_output/temp_image_descriptions.json"
        image_descriptions = describe_image(
            image_directory=output_images_folder,
            output_file=temp_json_path,
            model_name="llava:13b"
        )
    else:
        image_descriptions = {}
        temp_json_path = None
    
    with open(output_file, "a", encoding="utf-8") as txt_file:
        txt_file.write("\n\n=== EXTRACTED TABLES ===\n\n")
        if extracted_tables:
            for table_path in extracted_tables:
                txt_file.write(f"Table: {os.path.basename(table_path)}\n")
                with open(table_path, 'r', encoding='utf-8') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        txt_file.write(" | ".join(row) + "\n")
                txt_file.write("\n\n")
        else:
            txt_file.write("No tables found in the document.\n\n")
        
        # Add image descriptions
        txt_file.write("\n\n")
        if image_descriptions:
            for img_name, description in image_descriptions.items():
                txt_file.write(f"Image: {img_name}\n")
                txt_file.write(f"Description: {description}\n\n")
        else:
            txt_file.write("No images processed or no descriptions generated.\n\n")
    
    # Cleanup temporary files
    for image_path in extracted_image_paths:
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")
    
    for table_path in extracted_tables:
        try:
            os.remove(table_path)
            print(f"Deleted: {table_path}")
        except Exception as e:
            print(f"Error deleting {table_path}: {e}")
    
    if temp_json_path and os.path.exists(temp_json_path):
        try:
            os.remove(temp_json_path)
            print(f"Deleted: {temp_json_path}")
        except Exception as e:
            print(f"Error deleting {temp_json_path}: {e}")
    
    try:
        if os.path.exists(output_images_folder) and not os.listdir(output_images_folder):
            os.rmdir(output_images_folder)
            print(f"Removed empty directory: {output_images_folder}")
        
        if os.path.exists(output_tables_folder) and not os.listdir(output_tables_folder):
            os.rmdir(output_tables_folder)
            print(f"Removed empty directory: {output_tables_folder}")
    except Exception as e:
        print(f"Error removing directories: {e}")
    
    print(f"All data has been consolidated into {output_file}")

def process_all_pdfs(input_folder='./PDFinput'):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing {pdf_path}...")
            extract_pdf_data(pdf_path=pdf_path)

class PDFParser:
    def __init__(self, output_dir: str = "parser_output"):
        """Initialize the PDF parser."""
        self.output_dir = output_dir
        self.table_handler = TableHandler()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.images_dir = os.path.join(output_dir, "extracted_images")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF and identify thermodynamic properties."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_num, table in enumerate(page_tables, 1):
                            df = pd.DataFrame(table[1:], columns=table[0] if table[0] else [f"col_{i}" for i in range(len(table[0]))])
                            df.columns = [str(col).lower().strip() if col else f"col_{i}" for i, col in enumerate(df.columns)]
                            thermo_cols = []
                            for col in df.columns:
                                if any(term in col for term in ['temperature', 'temp', '°c', 'k', 'kelvin',
                                                             'melting', 'crystallization', 'force', 'energy']):
                                    thermo_cols.append(col)
                            
                            if thermo_cols:
                                table_name = f"thermo_table_{page_num}_{table_num}"
                                self.table_handler.store_table(df, table_name, {
                                    'source': pdf_path,
                                    'page': page_num,
                                    'table_num': table_num,
                                    'thermo_columns': thermo_cols
                                })
                                tables.append({
                                    'name': table_name,
                                    'columns': thermo_cols,
                                    'data': df.to_dict('records')
                                })
        except Exception as e:
            print(f"Error extracting tables: {e}")
        return tables

    def extract_images(self, pdf_path: str) -> List[str]:
        """Extract images from PDF and store them."""
        extracted_images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f"image_{page_num + 1}_{len(extracted_images) + 1}.{image_ext}"
                    image_path = os.path.join(self.images_dir, image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    extracted_images.append(image_path)

            if extracted_images:
                from LLAVA_image_description import describe_image
                descriptions = describe_image(
                    image_directory=self.images_dir,
                    output_file=os.path.join(self.output_dir, "image_descriptions.json"),
                    model_name="llava:13b"
                )
                
                for img_path, desc in descriptions.items():
                    self.table_handler.store_table(
                        pd.DataFrame({'description': [desc]}),
                        f"image_{img_path}",
                        {'type': 'image_description', 'source': pdf_path}
                    )
            
            return extracted_images
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []

    def process_pdf(self, pdf_path: str) -> str:
        """Process a PDF file and extract all relevant information."""
        tables = self.extract_tables(pdf_path)
        images = self.extract_images(pdf_path)
        doc = fitz.open(pdf_path)
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        doc.close()
        image_descriptions = {}
        if images:
            for img_path in images:
                img_name = os.path.basename(img_path)
                desc_df = self.table_handler.query_table(f"image_{img_name}")
                if not desc_df.empty:
                    image_descriptions[img_name] = desc_df['description'].iloc[0]
        
        output_file = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== PDF TEXT CONTENT ===\n\n")
            for i, text in enumerate(text_content, 1):
                f.write(f"--- Page {i} ---\n")
                f.write(text)
                f.write("\n\n")
            
            if tables:
                f.write("\n=== EXTRACTED TABLES ===\n\n")
                for table in tables:
                    f.write(f"Table: {table['name']}\n")
                    f.write("Thermodynamic columns: " + ", ".join(table['columns']) + "\n")
                    df = pd.DataFrame(table['data'])
                    f.write(df.to_string())
                    f.write("\n\n")
            
            if image_descriptions:
                f.write("\n=== EXTRACTED IMAGES AND DESCRIPTIONS ===\n\n")
                for img_name, description in image_descriptions.items():
                    f.write(f"Image: {img_name}\n")
                    f.write(f"Description: {description}\n")
                    f.write("\n")
        
        return output_file

    def close(self):
        """Close the database connection."""
        self.table_handler.close()

def main():
    """Main function to test the PDF parser."""
    parser = PDFParser()
    input_dir = "PDFinput"
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing {pdf_path}...")
            output_file = parser.process_pdf(pdf_path)
            print(f"Output saved to {output_file}")
    
    parser.close()

if __name__ == "__main__":
    main()
