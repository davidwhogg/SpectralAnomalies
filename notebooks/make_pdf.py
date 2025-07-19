#!/usr/bin/env python3

import os
from PIL import Image
import glob

def png_to_pdf(png_files, output_pdf="output.pdf"):
    """
    Convert a list of PNG images to a multi-page PDF.
    
    Args:
        png_files (list): List of PNG file paths
        output_pdf (str): Name of output PDF file
    """
    # Sort the provided files
    png_files = sorted(png_files)
    
    if not png_files:
        print("No PNG files provided")
        return
    
    print(f"Found {len(png_files)} PNG files")
    
    # Open all images
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            # Convert to RGB if necessary (PDFs don't support transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(img)
            print(f"Processed: {os.path.basename(png_file)}")
            
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
    
    if not images:
        print("No valid images to convert")
        return
    
    # Save as multi-page PDF
    try:
        images[0].save(
            output_pdf,
            format='PDF',
            save_all=True,
            append_images=images[1:],
            resolution=100.0
        )
        print(f"Successfully created {output_pdf} with {len(images)} pages")
        
    except Exception as e:
        print(f"Error creating PDF: {e}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <png_file1> [png_file2] ... <output_pdf>")
        print("Example: python script.py image1.png image2.png image3.png output.pdf")
        print("Example: python script.py *.png output.pdf  # (shell expands *.png)")
        sys.exit(1)
    
    # Last argument is the output PDF, everything else is input PNG files
    png_files = sys.argv[1:-1]
    output_file = sys.argv[-1]
    
    png_to_pdf(png_files, output_file)
