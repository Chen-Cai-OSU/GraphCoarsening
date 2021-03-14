import os
import yaml

indexes = [1, 2, 3]
indexes_str = [str(i) for i in indexes]
str = ','.join(indexes_str)
final = r'"\"' + str + r'\""'

dict = {'str':final}

output_txt = './data/test.txt'
# with open(output_txt, 'w') as file:
#     yaml.dump(dict, file, default_style=None)
c = None
with open(output_txt, 'r') as file:
    c = yaml.load(file, yaml.FullLoader)

for k, v in c.items():
    print(k, v)