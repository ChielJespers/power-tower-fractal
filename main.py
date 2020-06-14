from tqdm import tqdm
from PIL import Image, ImageDraw
import cmath

# ------------------------
# TO BE CUSTOMIZED BY USER
# ------------------------

# RENDERING PARAMETERS
sharpness  = 200                                                 # number of pixels specifying PNG pngWidth
maxIter    = 300                                                   # set higher for highly zoomed-in pictures

# COMPLEX DOMAIN
reStart = -4.15
reEnd = -4.09
imStart = -0.2
imEnd = 0.2
# ------------------------

# PNG DIMENSIONS; HEIGHT AUTOMATICALLY SCALES WITH WIDTH SPECIFIED BY 'sharpness' TO ENSURE NON-STRETCHED GRAPH
pngWidth = sharpness
pngHeight = int(pngWidth * (imEnd - imStart) / (reEnd - reStart))

# OPENING A BLACK-COLORED .PNG-FILE IN CORRECT DIMENSIONS
pic = Image.new('RGB', (pngWidth, pngHeight), (0, 0, 0))
draw = ImageDraw.Draw(pic)

for x in tqdm(range(0,pngWidth)):                                   # GENERATES A PROGRESS BAR WHILE RUNNING; ALSO GIVES ESTIMATE OF TIME REQUIRED
    for y in range(0, pngHeight):                                   # THIS IS A LEXICOGRAPHIC LOOP
        firstIterate = complex(
            reStart + (x / pngWidth) * (reEnd - reStart),           # SCALES CURRENT PNG-COORDINATE TO A COMPLEX NUMBER IN THE SPECIFIED DOMAIN
            imEnd - (y / pngHeight) * (imEnd - imStart))
        toggleOverflow = 0                                          # BECOMES 1 AFTER OVERFLOW ERROR, STOPPING THE WHILE-LOOP
        numberOfIterations = 0                                      # COUNTS THE NUMBER OF ITERATIONS ALREADY EXECUTED FOR CURRENT COMPLEX NUMBER
        if firstIterate == 0:
        	draw.point([x,y], (255,0,0))
        else:
            firstLog = cmath.log(firstIterate)
            nextIterate = firstIterate
            while numberOfIterations <= maxIter  and toggleOverflow == 0:
                try:
                    power = nextIterate * firstLog
                    nextIterate = cmath.exp(power)
                except OverflowError:
                    toggleOverflow = 1
                except ZeroDivisionError:
                    nextIterate = 0
                except Exception:
                    nextiterate = 1
                numberOfIterations += 1

        color = 255-int((numberOfIterations * 255 / maxIter))
        draw.point([x,y], (color,color,color))

pic.save('ZoomIslesharp.png', 'PNG')