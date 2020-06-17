from tqdm import tqdm
from PIL import Image, ImageDraw

MAX_ITER = 20000
PRECISION = 3  # digits
CONV_RADIUS = 10 ** (-PRECISION)

# COMPLEX DOMAIN
RE_START = -4
RE_END = 5
IM_START = -4
IM_END = 4

# SHARPNESS
WIDTH = 20
HEIGHT = int(WIDTH * (IM_END - IM_START) / (RE_END - RE_START))

palette = []

im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0)) # alles 0 is zwart, alles 255 is wit
draw = ImageDraw.Draw(im)

for x in tqdm(range(0,WIDTH)):
    for y in range(0, HEIGHT):
        c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                    IM_END - (y / HEIGHT) * (IM_END - IM_START))
        OFE = 0
        n= 0
        e= 1
        z= c
        while n < MAX_ITER and e > CONV_RADIUS and OFE == 0:
            try:
                z = c ** z
                w = c ** z
                e = abs(w - z)
            except OverflowError as E:
                OFE = 1
            n += 1

        if e < CONV_RADIUS:
            draw.point([x, y], (0,0,0))
        elif n == MAX_ITER:
            k = 1
            while abs(z - w) > CONV_RADIUS and k < MAX_ITER + 1:
                w = c ** w
                k += 1
            if k == 2:
                draw.point([x, y], (255,0,0)) #red
            elif k == 3:
                draw.point([x, y], (0,255,0)) #green
            elif k == 4:
                draw.point([x, y], (0, 0, 255)) #blue
            elif k == 5:
                draw.point([x, y], (255, 255, 0)) #yellow
            elif k == 6:
                draw.point([x, y], (127, 0, 255)) #purple
            elif k == 7:
                draw.point([x, y], (255, 128, 0)) #orange
            elif k == MAX_ITER:
                draw.point([x, y], (0,0,0)) # assumed convergence
            else:
                draw.point([x, y], (127, 127, 127))  # gray
        else:
            draw.point([x, y], (255,255,255))

im.save('output.png', 'PNG')