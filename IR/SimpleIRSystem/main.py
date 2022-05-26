if __name__ == '__main__':
    print('PyCharm')

import os

print(os.environ.get('JAVA_HOME'))

import analyze as an

txt = an.fileToString('example.txt')
print(an.tagMe(txt))