import os
import shutil
from random import shuffle
from math import floor

def get_file_list_from_dir(datadir):
	all_files = os.listdir(os.path.relpath(datadir))
	data_files = list(filter(lambda file: file.endswith('.png'), all_files))
	return data_files

def randomize_files(file_list):
	shuffle(file_list)
	return file_list

def get_training_and_testing_sets(file_list):
	split = 0.7
	split_index = floor(len(file_list) * split)
	training = file_list[:split_index]
	testing = file_list[split_index:]
	return training, testing

def main():
	datadir = "./data"
	data_files = get_file_list_from_dir(datadir)
	randomized = randomize_files(data_files)
	training, testing = get_training_and_testing_sets(randomized)
	for f in training:
		shutil.move('./data/' + f, "./train/")
	for f in testing:
		shutil.move('./data/' + f, "./test/")

if __name__ == "__main__":
	main()
