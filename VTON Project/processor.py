import sys
import io
import shutil
import os
from gradio_client import Client, handle_file

# --- WINDOWS ENCODING FIX ---
# REQUIRED to prevent crashes when gradio_client prints status symbols (like ✔)
# You must include these lines if you are running on Windows/Command Prompt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ----------------------------

# --- CONFIG ---
INPUT_FOLDER = "captured_poses"
OUTPUT_FOLDER = "final_output"
CLOTH_IMAGE = "tshirt.jpg" 

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

def run_cloud_physics():
    if not os.path.exists(CLOTH_IMAGE):
        print(" ERROR: 'tshirt.jpg' is missing!")
        current_dir = os.getcwd()
        print(f" -> I am looking in this folder: {current_dir}")
        return

    print("--- Connecting to AI Cloud Engine ---")
    
    # This is the free public API
    client = Client("yisol/IDM-VTON") 
    
    views = ["front.jpg", "right_side.jpg", "left_side.jpg"]
    
    for view in views:
        person_path = os.path.join(INPUT_FOLDER, view)
        if not os.path.exists(person_path):
            print(f" Skipping {view} (Not found)")
            continue
            
        print(f" Processing {view}... (This takes ~30-60 seconds)")
        
        try:
            result = client.predict(
                dict={"background": handle_file(person_path), "layers": [], "composite": None},
                garm_img=handle_file(CLOTH_IMAGE),
                garment_des="t-shirt",
                is_checked=True, 
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            
            # Result[0] is the final image path in the temp folder
            final_path = os.path.join(OUTPUT_FOLDER, f"dressed_{view}")
            shutil.move(result[0], final_path)
            print(f" Saved: {final_path}")
            
        except Exception as e:
            print(f" Error processing {view}: {e}")

if __name__ == "__main__":
    run_cloud_physics()