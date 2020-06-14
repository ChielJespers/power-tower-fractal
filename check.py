from PIL import Image, ImageDraw

RE_START = -100
RE_END = 100
IM_START = -100
IM_END = 100

WIDTH = 10000
HEIGHT = int(WIDTH * (IM_END - IM_START) / (RE_END - RE_START)) + 1

MAX_ITER = 2000
PRECISION = 3 #digits
CONV_RADIUS = 10**(-PRECISION)

REAL = 1
IMAG = 2

c = complex(REAL,IMAG)

z = c
n = 0
e = 1
OFE = 0

im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0)) # alles 0 is zwart, alles 255 is wit
draw = ImageDraw.Draw(im)

while n < MAX_ITER and e > CONV_RADIUS and OFE == 0:
    try:
        z = c**z
        w = c**z
        e = abs(w-z)
    except OverflowError as E:
        OFE = 1
    n += 1
    print(z,e)

    x = round(z.real * WIDTH)
    y = HEIGHT - round(z.imag * HEIGHT)
    gradient = round(min(255,n)*3)
    draw.point([x, y], (gradient,0,255-gradient))

print("--------------------------------------------")

if e < CONV_RADIUS:
    l = complex(round(z.real,PRECISION),round(z.imag,PRECISION))
    print("Convergence to", l, "after", n, "iterations.")
elif n == MAX_ITER:
    print("The sequence converges to the following cycle:")
    k = 1
    print(w)
    while abs(z-w) > CONV_RADIUS**2:
        w = c**w
        k += 1
        print(w)
    print("The cycle contains", k, "fixed points.")
else:
    print("Diverges to infinity; overflows after", n, "iterations")

im.save('orbit.png','PNG')