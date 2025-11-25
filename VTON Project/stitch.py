import cv2
import os

def stitch_images(img_list, output_filename="stitched_panorama.jpg"):
    """
    Stitches a list of images together using the OpenCV Stitcher class.
    
    Args:
        img_list (list): A list of OpenCV image objects (loaded using cv2.imread).
        output_filename (str): The name for the saved output file.
    """
    if len(img_list) < 2:
        print("Need at least two images to stitch.")
        return

    print(f"Starting stitch process for {len(img_list)} images...")
    
    # 1. Create the Stitcher object
    # cv2.Stitcher_PANORAMA is a general-purpose stitching mode
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    
    # 2. Perform the stitching
    # 'status' is a code indicating success or failure.
    # 'pano' is the resulting stitched image.
    status, pano = stitcher.stitch(img_list) 
    
    if status == cv2.Stitcher_OK:
        print("[SUCCESS] Images stitched successfully.")
        
        # 3. Save the result
        cv2.imwrite(output_filename, pano)
        print(f"[OUTPUT] Saved final image to: {output_filename}")
        return pano
    else:
        # Handle different error codes (e.g., ERR_NEED_MORE_IMGS, ERR_HOMOGRAPHY_EST_FAIL)
        print(f"[FAILURE] Image stitching failed. Status code: {status}")
        print("This often happens if there's not enough overlap or features between the images.")
        return None

if __name__ == "__main__":
    # Define your image paths
    # Assuming you have captured: front.jpg, right_side.jpg, left_side.jpg
    IMAGE_DIR = "captured_poses"
    
    # --- Stitching Strategy: Stitch Left + Front, then result + Right ---
    
    # A. Load the first two images (e.g., the front and one side)
    img_1_path = os.path.join(IMAGE_DIR, "front.jpg")
    img_2_path = os.path.join(IMAGE_DIR, "right_side.jpg")
    
    # Check if files exist and are readable before loading
    if not os.path.exists(img_1_path) or not os.path.exists(img_2_path):
        print("Error: Required images not found.")
    else:
        img_1 = cv2.imread(img_1_path)
        img_2 = cv2.imread(img_2_path)

        if img_1 is not None and img_2 is not None:
            # B. Run the stitching process
            stitched_result = stitch_images([img_1, img_2], "temp_stitched_result.jpg")
            
            # You would repeat this process to add the third image to the 'stitched_result'
            # (e.g., stitch_images([stitched_result, img_3], "final_3_side_view.jpg"))