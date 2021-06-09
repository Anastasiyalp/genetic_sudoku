import re
import matplotlib.pyplot as plt
import numpy
import os
from collections import OrderedDict
f = open("./Test20/" + "check.txt", "r")

results = []
# difficalty = {'1': ['s01a.txt', 's01b.txt', 's01c.txt'], '2': ['s02a.txt', 's02b.txt', 's02c.txt'],
#               '3': ['s03a.txt', 's03b.txt', 's03c.txt'], '4': ['s04a.txt', 's04b.txt', 's04c.txt'],
#               '5': ['s05a.txt', 's05b.txt', 's05c.txt'], 'E': ['s06a.txt', 's06b.txt', 's06c.txt'],
#               'C': ['s07a.txt', 's07b.txt', 's07c.txt'], 'D': ['s08a.txt', 's08b.txt', 's08c.txt'],
#               'SD': ['s09a.txt', 's09b.txt', 's09c.txt'], 'Easy': ['s10a.txt', 's10b.txt', 's10c.txt'],
#               'Medium': ['s11a.txt', 's11b.txt', 's11c.txt'], 'Hard': ['s12a.txt', 's12b.txt', 's12c.txt'],
#               'GA-E': ['s13a.txt', 's13b.txt', 's13c.txt'], 'GA-M': ['s14a.txt', 's14b.txt', 's14c.txt'],
#               'GA-H': ['s15a.txt', 's15b.txt', 's15c.txt'], 'AI Escargot': ['s16.txt']}
difficaltys = ['1','1','1','2','2','2','3','3','3','4','4','4','5','5','5','E','E','E','C','C','C','D','D','D',
               'SD','SD','SD','Easy','Easy','Easy','Medium','Medium','Medium','Hard','Hard','Hard',
               'GA-E','GA-E','GA-E','GA-M','GA-M','GA-M','GA-H','GA-H','GA-H','AI Escargot']
for line in f.readlines():
    file, generation, time, launch = re.match("(\w+\.txt) \: generations \= (\d+) time \= (\d+\.\d+) launchs \= (\d+)", line).groups()
    results.append([file, int(generation), int(float(time))])

sudoku_numbers = dict()
filenames = os.listdir("./sudoku")
print filenames
# for filename in filenames:
#     count = 0
#     for x in numpy.loadtxt("./sudoku/" + filename).reshape(81).astype(int):
#         if x == 0:
#             count += 1
#     sudoku_numbers[str(filename)] = count
#
# sudoku_numbers = OrderedDict(sorted(sudoku_numbers.items(), key=lambda item: item[1]))
# print sudoku_numbers


def union(result):
    union_result = [result[0][0], 0, 0]
    for el in result:
        union_result[1] += el[1]
        union_result[2] += el[2]
    return union_result


# values = set(map(lambda x:x[0], results))
format_result = map(union, [[y for y in results if y[0]==x] for x in filenames])

for x in format_result:
    x[1] /= 4
    x[2] /= 4
print format_result

format_result = numpy.array(format_result).swapaxes(0, 1)

# print format_result[0],
# print numpy.array(format_result[1]).astype(int)/20
# print numpy.array(format_result[2]).astype(int)/20

# print format_result

plt.figure(figsize=(8, 6), dpi=80)
# print sudoku_numbers


# plt.plot(sudoku_numbers.values(), numpy.array(format_result[1]).astype(int)/20, 'go', label="numbers")
# plt.plot(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int)/20, 'b*', label="generations")
# plt.plot(numpy.array(format_result[1]).astype(int)/20, sudoku_numbers.values(), 'go', label="generations")
# plt.plot(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values(), 'b*', label="times")



# plt.scatter(sudoku_numbers.values(), numpy.array(format_result[1]).astype(int)/4,
#             c=numpy.arctan2(sudoku_numbers.values(), numpy.array(format_result[1]).astype(int)/4), cmap='winter', s=50, alpha=0.8)


# plt.scatter(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int)/20,
#             c=numpy.arctan2(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int)/20), cmap='spring', s=50, alpha=0.8)
# plt.scatter(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values(),
#             c=numpy.arctan2(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values()), cmap='winter', s=50, alpha=0.8)
# plt.scatter(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values(), 'b*', label="times")

# plt.plot(sudoku_numbers.values(), label="numbers", color="red", linewidth=1.0, linestyle="-")
# plt.plot(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int))
# plt.plot(sudoku_numbers.values(), format_result[2])

plt.scatter(difficaltys, numpy.array(format_result[1]).astype(int))
# plt.ylim((0, 1000))
# plt.scatter(sudoku_numbers.values(), numpy.array(format_result[1]).astype(int)/numpy.array(format_result[2]).astype(float), c=numpy.arctan2(sudoku_numbers.values(), numpy.array(format_result[1]).astype(int)/numpy.array(format_result[2]).astype(float)), cmap='spring', s=50, alpha=0.8)
# plt.scatter(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int)/numpy.array(format_result[1]).astype(float), c=numpy.arctan2(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int)/numpy.array(format_result[1]).astype(float)), cmap='spring', s=50, alpha=0.8)
# plt.plot(numpy.array(format_result[1]).astype(int)/numpy.array(format_result[2]).astype(float), label="generations", color="blue", linewidth=1.0, linestyle="-")
# plt.plot(numpy.array(format_result[2]).astype(float)/20, label="times", color="black", linewidth=1.0, linestyle="-")
# plt.plot(numpy.array(format_result[2]).astype(float), label="times", color="red", linewidth=1.0, linestyle="-")
plt.ylabel('Generations')
plt.xlabel('Rate of difficult')
# plt.xlim(0, len(best_data))
# plt.legend()
plt.show()
# plt.savefig("./results("+ file +")/"+str(times)+".png")