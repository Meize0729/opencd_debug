bx = []
cd_list = []

with open('/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_bx.txt', 'w') as fbx:
    with open('/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_cd.txt', 'w') as fcd:
        with open('/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_nooverlap.txt', 'r') as fr:
            for line in fr.readlines():
                a, b, cd, label_a, label_b = line.strip().split('\t')
                if cd not in cd_list:
                    fcd.write(f'{a}\t{b}\t{cd}\t**\t**\n')
                if a not in bx:
                    bx.append(a)
                    fbx.write(f'{a}\t**\t**\t{label_a}\t**\n')
                if b not in bx:
                    bx.append(b)
                    fbx.write(f'{b}\t**\t**\t{label_b}\t**\n')

# dada = []
# with open('/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_nooverlap.txt', 'r') as fr:
#     for line in fr.readlines():
#         a, b, cd, label_a, label_b = line.strip().split('\t')
#         if cd in dada:
#             print(cd)
#         else:
#             dada.append(cd)
# print(len(dada))