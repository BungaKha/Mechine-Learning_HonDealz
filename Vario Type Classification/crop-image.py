import os
from PIL import Image
import numpy as np

def crop_to_square(image_path, output_path, target_size=224):
    """
    Crop image to square while keeping the motorcycle centered
    """
    try:
        # open image
        img = Image.open(image_path)
        
        # convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # get dimensions
        width, height = img.size
        
        # calculate dimensions for center crop
        if width > height:
            left = (width - height) // 2
            right = left + height
            top = 0
            bottom = height
        else:
            top = (height - width) // 2
            bottom = top + width
            left = 0
            right = width
        
        # crop iamge to square
        img_cropped = img.crop((left, top, right, bottom))
        
        # resize to target size
        img_resized = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # save processed image
        img_resized.save(output_path, 'JPEG', quality=95)
        return True
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_dataset(input_base_dir, output_base_dir, target_size=224):
    """
    Process entire dataset maintaining directory structure
    """
    # create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # counter for processed images
    total = 0
    successful = 0
    
    # walk through directory
    for root, dirs, files in os.walk(input_base_dir):
        # create corresponding output directory
        rel_path = os.path.relpath(root, input_base_dir)
        output_dir = os.path.join(output_base_dir, rel_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # process each image
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.jpg")
                
                if crop_to_square(input_path, output_path, target_size):
                    successful += 1
                
                if total % 10 == 0:
                    print(f"Processed {total} images. Success rate: {successful/total*100:.2f}%")
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Success rate: {successful/total*100:.2f}%")

input_dataset = "D:\Bangkit\capstone\dataset"  # current dataset path
output_dataset = "D:/Bangkit/capstone/clean"  # where to save processed images

# process both train and validation sets
for subset in ['train', 'validation']:
    input_dir = os.path.join(input_dataset, subset)
    output_dir = os.path.join(output_dataset, subset)
    print(f"\nProcessing {subset} set...")
    process_dataset(input_dir, output_dir)