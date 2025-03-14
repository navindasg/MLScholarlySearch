import os
import fitz
import camelot

def extract_pdf_data(pdf_path='./PDFinput/Efficient-perovskite-light-emitting-diodes-featuring-nanometre-sized-crystallites.pdf', 
                     output_text_file='./parser_output/output_text.txt', 
                     output_images_folder='./parser_output/extracted_images', 
                     output_tables_folder='./parser_output/extracted_tables',
                     image_area_threshold=1000):
    """
    Extracts text, images, and tables from a PDF file.
    
    Parameters:
      pdf_path (str): Path to the PDF file.
      output_text_file (str): Path to the output text file.
      output_images_folder (str): Directory where extracted images will be saved.
      output_tables_folder (str): Directory where extracted tables (CSV files) will be saved.
      image_area_threshold (int): Minimum area (width x height in pixels) required to save an image.
                                  Images below this area will be skipped.
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
