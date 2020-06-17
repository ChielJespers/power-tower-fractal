all:
	nvcc  -o powtowfrac linzoom.cu -lgd -lm -ldl