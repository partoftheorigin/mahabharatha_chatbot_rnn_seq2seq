import os
import re

def pre_process_subtitles(dir):
    fh = open('pre_spaces.txt', mode='w')
    for file in os.listdir(dir):
        f = open((dir + '/' + file), encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.encode('ascii', 'ignore').decode('ascii')
            if re.search('[a-zA-Z]', line):
                clean_line = line.strip("'\n'")
                fh.write(clean_line+'\n')
    return print('Subtitles Cleaned!')


def


if __name__ == '__main__':
    pre_process_subtitles(os.getcwd() + '/dataset/mahabharat-subtitles')