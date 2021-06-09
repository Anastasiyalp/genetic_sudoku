import re
import matplotlib.pyplot as plt
import numpy
import os
from collections import OrderedDict
f = open("./Test20/" + "check.txt", "r")

results = []
for line in f.readlines():
    file, generation, time, launch = re.match("(\w+\.txt) \: generations \= (\d+) time \= (\d+\.\d+) launchs \= (\d+)", line).groups()
    results.append([file, int(generation), int(float(time)), int(launch)])

def union(result):
    union_result = [result[0][0], 0, 0, 0]
    for el in result:
        union_result[1] += el[1]
        union_result[2] += el[2]
        union_result[3] += el[3]
    return union_result


values = map(lambda x:x[0], results)
format_result = map(union, [[y for y in results if y[0]==x] for x in values])
format_result = numpy.array(format_result).swapaxes(0, 1)

plt.figure(figsize=(16, 6))


# plt.plot(sudoku_numbers.values(), numpy.array(format_result[1]).astype(int)/20, 'go', label="numbers")
# plt.plot(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int)/20, 'b*', label="generations")
# plt.plot(numpy.array(format_result[1]).astype(int)/20, sudoku_numbers.values(), 'go', label="generations")
# plt.plot(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values(), 'b*', label="times")
# plt.scatter(values, numpy.array(format_result[1]).astype(int)/20,
#             c=numpy.arctan2(values, numpy.array(format_result[1]).astype(int)/20), cmap='winter', s=50, alpha=0.8)
plt.plot(values, numpy.array(format_result[1]).astype(int)/20)
# plt.scatter(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values(),
#             c=numpy.arctan2(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values()), cmap='winter', s=50, alpha=0.8)
# plt.scatter(numpy.array(format_result[2]).astype(int)/20, sudoku_numbers.values(), 'b*', label="times")

# plt.plot(sudoku_numbers.values(), label="numbers", color="red", linewidth=1.0, linestyle="-")
# plt.plot(sudoku_numbers.values(), numpy.array(format_result[2]).astype(int))
# plt.plot(sudoku_numbers.values(), format_result[2])
# plt.plot(numpy.array(format_result[1]).astype(int), label="generations", color="blue", linewidth=1.0, linestyle="-")
# plt.plot(numpy.array(format_result[2]).astype(float), label="times", color="black", linewidth=1.0, linestyle="-")
# plt.plot(numpy.array(format_result[2]).astype(float), label="times", color="red", linewidth=1.0, linestyle="-")
plt.ylabel('Generation')
# plt.xlim(0, len(best_data))
# plt.legend()
plt.show()
# plt.savefig("./results("+ file +")/"+str(times)+".png")