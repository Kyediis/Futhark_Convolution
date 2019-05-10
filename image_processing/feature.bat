python blur-png.py image.png --kernel sharp_filter.in --output-file image-sharpened.png
python blur-png.py image-sharpened.png --kernel edge_filter.in --output-file image-edge.png
python blur-png.py image-edge.png --kernel blur_filter.in --output-file image-blurred.png