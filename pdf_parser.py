import os
import fitz
import camelot
from PIL import Image
import io
import numpy as np
import cv2
import statistics

def extract_pdf_data(pdf_path='./PDFinput/Efficient-perovskite-light-emitting-diodes-featuring-nanometre-sized-crystallites.pdf', 
                     output_text_file='./parser_output/output_text.txt', 
                     output_images_folder='./parser_output/extracted_images', 
                     output_tables_folder='./parser_output/extracted_tables',
                     image_area_threshold=1000,
                     color_similarity_threshold=0.9):
    """
    Extracts text, images, and tables from a PDF file.
    
    Parameters:
      pdf_path (str): Path to the PDF file.
      output_text_file (str): Path to the output text file.
      output_images_folder (str): Directory where extracted images will be saved.
      output_tables_folder (str): Directory where extracted tables (CSV files) will be saved.
      image_area_threshold (int): Minimum area (width x height in pixels) required to save an image.
                                  Images below this area will be skipped.
      color_similarity_threshold (float): Maximum percentage of pixels allowed to be similar to the most dominant color.
                                         Value between 0.0 and 1.0. Images above this threshold will be skipped.
    """
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_tables_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    #Text Extraction
    with open(output_text_file, "w", encoding="utf-8") as txt_file:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            txt_file.write(f"--- Page {page_num + 1} ---\n")
            txt_file.write(text)
            txt_file.write("\n\n")
    print("Text extraction complete. Saved to", output_text_file)
    
    #Image Extraction
    image_counter = 1
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
            import numpy as np
            import cv2
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
            print(f"Extracted image saved as {image_filename}")
            image_counter += 1
    print("Image extraction complete. Saved in folder", output_images_folder)
    
    #Table Extraction
    try:
        tables = camelot.read_pdf(pdf_path, pages='all')
        if tables:
            for i, table in enumerate(tables):
                table_csv_path = os.path.join(output_tables_folder, f"table_{i + 1}.csv")
                table.to_csv(table_csv_path)
                print(f"Extracted table saved as {table_csv_path}")
            print("Table extraction complete. Saved in folder", output_tables_folder)
        else:
            print("No tables found in the PDF.")
    except Exception as e:
        print("Error extracting tables:", e)

if __name__ == '__main__':
    extract_pdf_data()
