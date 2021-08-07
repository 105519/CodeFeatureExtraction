'''
Calls the ENRE tool for each item in the folder 'py150_files'，
and saves the output of ENRE to the folder 'py150_structures'
'''

import os
import time

total = 0
os.system('rm py150_structures -r')
assert os.path.isdir('py150_files/data')
assert os.system('mkdir py150_structures') == 0
for dir1 in os.listdir('py150_files/data'):
    assert os.path.isdir('py150_files/data/' + dir1)
    assert os.system('mkdir py150_structures/' + dir1) == 0
    for dir2 in os.listdir('py150_files/data/' + dir1):
        assert os.path.isdir('py150_files/data/' + dir1 + '/' + dir2)
        assert os.system('mkdir py150_structures/' + dir1 + '/' + dir2) == 0
        total += 1
print(total, ' projects in total')

startTime = time.time()
cnt = 0
failList = []
for dir1 in os.listdir('py150_files/data'):
    for dir2 in os.listdir('py150_files/data/' + dir1):
        cnt += 1
        print('%4d/%d started: using time = %.2lf (%s / %s)' % (cnt, total, time.time() - startTime, dir1, dir2))
        if (os.system('java -jar ../tools/ENRE/ENRE-v1.0.jar python ' + 'py150_files/data/' + dir1 + '/' + dir2 + ' null tmp' + ' >/dev/null 2>&1') == 0):
            assert(os.system('mv tmp-out/tmp_node.csv py150_structures/' + dir1 + '/' + dir2 + '/' + 'node.txt') == 0)
            assert(os.system('mv tmp-out/tmp_edge.csv py150_structures/' + dir1 + '/' + dir2 + '/' + 'edge.txt') == 0)
        else:
            assert(os.system('rm py150_structures/' + dir1 + '/' + dir2 + ' -r') == 0)
            failList.append([cnt, dir1 + '/' + dir2])
        assert(os.system('rm tmp-out -r') == 0)
print('failList =', failList)
