'''
Checks the number of files in each folder in raw data and if all files are mp3.
'''
import os
from config.config import RAW_DATA_DIR

for category in os.listdir(RAW_DATA_DIR):
    files = os.listdir(f"{RAW_DATA_DIR}/{category}")
    print(f"{category}: {len(files)} files")
    count_not_mp3 = 0
    for file in files:
        if (file.split(".")[-1] != 'mp3'):
            count_not_mp3 += 1
    if count_not_mp3 != 0:
        print(f"\033[91m{count_not_mp3} files are of different type\033[0m")
    else:
        print("\033[92mAll files are mp3\033[0m")