python conv-png.py images/image.png --kernel sobelx.in --output-file images/sobelx.png
python conv-png.py images/image.png --kernel sobely.in --output-file images/sobely.png
python mag-png.py images/sobelx.png images/sobely.png --output-file images/sobel_edge.png