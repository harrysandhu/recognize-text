# Recognize Text
- Built with opencv.

- The problem
- take an image:
    - use EAST detector model
    - Conv_7/Sigmoid layer gives the probability of region containing text or not.
    - concat_3 outputs a feature map that gives the geometry of the bounding box.
    - detect the text
    - create a bounding box over the text
        - x, y, size
    - recognize the text
    - output the bounding box with the image
     
- Kinda sucky for videos, if you good with this, help improve this.
 
# TIP:
- Use google's MLKit for ocr type stuff, because it has better pipeline. 
