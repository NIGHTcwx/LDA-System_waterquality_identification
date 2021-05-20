import os
import sys

file1path = 'train_labels.txt'
file2path = 'a1.txt'
file3path = 'a2.txt'
file4path = 'a3.txt'

file_1 = open(file1path, 'r')
file_2 = open(file2path, 'r')
file_3 = open(file3path, 'r')
file_4 = open(file4path, 'r')

list1 = []
for line in file_1.readlines():
    ss = line.strip()
    list1.append(ss)
file_1.close()

list2 = []
for line in file_2.readlines():
    ss = line.strip()
    list2.append(ss)
file_2.close()

list3 = []
for line in file_3.readlines():
    ss = line.strip()
    list3.append(ss)
file_3.close()

list4 = []
for line in file_4.readlines():
    ss = line.strip()
    list4.append(ss)
file_4.close()

file_new = open('train_lda.txt', 'w')
for i in range(len(list1)):
    sline = list1[i] + ' ' + list2[i] + ' ' + list3[i] + ' ' + list4[i]
    file_new.write(sline+'\n')
file_new.close()

# 文本排序并保存
output_file = 'train_lda.txt'
output_file_sort = 'train_lda_sort.txt'
with open(output_file_sort, 'a') as writersort:
    writersort.write(''.join(sorted(open(output_file), key=lambda s: s.split('\t')[0])))


