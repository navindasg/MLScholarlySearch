import os
import shutil

def clear_debug_logs():
    debug_dir = "./debug_logs"
    
    # Check if directory exists
    if not os.path.exists(debug_dir):
        print(f"Debug logs directory '{debug_dir}' does not exist.")
        return
    
    try:
        # Remove all files in the directory
        for filename in os.listdir(debug_dir):
            file_path = os.path.join(debug_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        print(f"Successfully cleared all files in '{debug_dir}'")
        
    except Exception as e:
        print(f"Error clearing debug logs: {e}")

if __name__ == "__main__":
    clear_debug_logs() 
