all:
	nvcc  -o powtowfrac main.cu -lgd -lm -ldl