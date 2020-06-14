from tqdm import tqdm
from PIL import Image, ImageDraw

# ------------------------
# TO BE CUSTOMIZED BY USER
# ------------------------

# RENDERING PARAMETERS
sharpness  = 500                                                    # number of pixels specifying PNG pngWidth
maxIter    = 30000                                                  # set higher for highly zoomed-in pictures
precision  = 3                                                      # measure of precision (see convergenceRadius below)

# COMPLEX DOMAIN
reStart = -2
reEnd = 0
imStart = 0
imEnd = 1
# ------------------------

# LIST OF COLORS USED IN SPECIFIC ORDER
colorList = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
             (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
             (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
             (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128)]

# DEFINE EPSILON TO ASSUME CAUCHY CONVERGENCE WHEN DIFFERENCE BECOMES SMALLER THAN THIS (CURRENTLY 'precision' is NUMBER OF DIGITS)
convergenceRadius = 10 ** (-precision)

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
        difference = convergenceRadius + 1                          # KEEPS TRACK OF ABSOLUTE DIFFERENCE BETWEEN CURRENT AND NEXT ITERATION
        nextIterate = firstIterate                                                                                        
        while numberOfIterations < maxIter and difference > convergenceRadius and toggleOverflow == 0:
            try:
                nextIterate = firstIterate ** nextIterate
                nextNextIterate = firstIterate ** nextIterate
                difference = abs(nextNextIterate - nextIterate)
            except OverflowError as E:
                toggleOverflow = 1
            except ZeroDivisionError as D:
                difference = 0

        if difference < convergenceRadius:
            draw.point([x, y], (0,0,0))
        elif numberOfIterations == maxIter:
            cycleLength = 1
            while abs(nextIterate - nextNextIterate) > convergenceRadius and cycleLength <= maxIter:
                nextNextIterate = firstIterate ** nextNextIterate
                cycleLength += 1
            if cycleLength <= 19:
                draw.point([x, y], colorList[cycleLength-2])
            else:
                draw.point([x, y], (127, 127, 127))  # gray
        else:
            draw.point([x, y], (255,255,255))

pic.save('output.png', 'PNG')