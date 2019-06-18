from PIL import Image
import os
import sys

def main():
	arg1 = str(sys.argv[1])
	arg2 = str(sys.argv[2])
	directory = os.fsencode(arg1)
	count = 0
	for file in os.listdir(arg1):
		if file.endswith(".png"):
			im = Image.open(file)
			im = im.convert('RGB')
			im.save(arg2 + "/IMG_" + str(count) + ".jpg", quality=100)
			count += 1

if __name__ == "__main__":
    main()
