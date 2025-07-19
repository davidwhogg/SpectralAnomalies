import os
from PIL import Image
import glob

def png_to_pdf(input_filespec, output_pdf="output.pdf"):
    """
    Convert PNG images matching a filespec to a multi-page PDF.
    
    Args:
        input_filespec (str): File specification with wildcards (e.g., "*.png", "images/page_*.png")
        output_pdf (str): Name of output PDF file
    """
    # Get all PNG files matching the filespec and sort them
    png_files = sorted(glob.glob(input_filespec))
    
    if not png_files:
        print(f"No files found matching pattern: {input_filespec}")
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
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_filespec> <output_pdf>")
        print("Example: python script.py '*.png' combined_images.pdf")
        print("Example: python script.py 'images/page_*.png' report.pdf")
        print("Example: python script.py '/path/to/files/img_[0-9][0-9].png' numbered.pdf")
        sys.exit(1)
    
    input_filespec = sys.argv[1]
    output_file = sys.argv[2]
    
    png_to_pdf(input_filespec, output_file)
