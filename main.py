import numpy
import random
import os
import sys
import time

random.seed()
Nd = 9


class Population(object):
    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []

        given_columns = numpy.copy(given).swapaxes(0, 1)
        given_sub_blocks = numpy.copy(given).reshape((3, 3, 3, 3)).transpose((0, 2, 1, 3)).reshape((9, 9))

        full_set = set(range(1, Nd + 1))
        missed_rows = []
        missed_columns = []
        missed_sub_blocks = []

        for i in range(0, Nd):
            missed_rows.append((full_set - set(given[i])))
            missed_columns.append((full_set - set(given_columns[i])))
            missed_sub_blocks.append((full_set - set(given_sub_blocks[i])))

        helper = numpy.zeros((Nd, Nd), dtype=list)
        for i in range(0, Nd):
            for j in range(0, Nd):
                if given[i][j] != 0:
                    helper[i][j] = [given[i][j]]
                else:
                    helper[i][j] = list(missed_rows[i] & missed_columns[j] & missed_sub_blocks[(i / 3) * 3 + j / 3])

        for p in range(0, Nc):
            self.candidates.append(Candidate(helper))
            for i in range(0, Nd):
                to_fill = []
                for ind, h in enumerate(helper[i]):
                    if len(h) == 1:
                        self.candidates[p].values[i][ind] = h[0]
                    else:
                        to_fill.append(ind)
                while full_set - set(self.candidates[p].values[i]):
                    for j in to_fill:
                        self.candidates[p].values[i][j] = random.choice(helper[i][j])
        self.update_fitness()
        print("Seeding complete.")
        return

    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return

    def sort(self):
        self.candidates.sort(key=lambda b: -b.fitness)
        return


class Candidate(object):
    def __init__(self, helper):
        self.values = numpy.zeros((Nd, Nd), dtype=int)
        self.fitness = None
        self.helper = helper
        return

    def update_fitness(self):
        column_values = numpy.copy(self.values).swapaxes(0,1)
        column_sum = 0
        sub_block_sum = 0
        sub_blocks_values = numpy.copy(self.values).reshape((3, 3, 3, 3)).transpose((0, 2, 1, 3)).reshape((9, 9))

        for i in range(0, Nd):
            column_sum += (1.0 / (9-len(set(column_values[i]))+1)) / Nd
            sub_block_sum += (1.0 / (9-len(set(sub_blocks_values[i]))+1)) / Nd

        if int(column_sum) == 1 and int(sub_block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * sub_block_sum

        self.fitness = fitness
        return

    def mutate(self, mutation_rate, given):
        r = random.uniform(0, 1.0)

        success = False
        if r < mutation_rate:
            while not success:
                row = random.randint(0, 8)

                available = [ind for ind, g in enumerate(given[row]) if g == 0]
                random.choice(available)

                from_column = random.choice(available)
                available.remove(from_column)
                to_column = random.choice(available)

                if (self.values[row][from_column] in self.helper[row][to_column] and
                        self.values[row][to_column] in self.helper[row][from_column]):
                    self.values[row][to_column], self.values[row][from_column] = self.values[row][from_column], self.values[row][to_column]
                    success = True

    def compete(self, competitor):
        if self.fitness > competitor.fitness:
            fittest = self
            weakest = competitor
        else:
            fittest = competitor
            weakest = self

        selection_rate = 0.85
        if random.uniform(0, 1.0) < selection_rate:
            return fittest
        else:
            return weakest

    def crossover(self, parent2, crossover_rate):
        child1 = Candidate(self.helper)
        child2 = Candidate(self.helper)

        child1.values = numpy.copy(self.values)
        child2.values = numpy.copy(parent2.values)

        if random.uniform(0, 1.0) < crossover_rate:
            crossover_point1 = 0
            crossover_point2 = 0
            while crossover_point1 == crossover_point2:
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(0, 8)

            for i in range(min(crossover_point1, crossover_point2), max(crossover_point1, crossover_point2)+1):
                child1.values[i], child2.values[i] = child2.values[i], child1.values[i]
        return child1, child2


class Sudoku(object):

    def __init__(self):
        self.population = Population()
        self.given = None
        self.best_data = []
        return

    def load(self, path):
        with open(path, "r") as f:
            self.given = numpy.loadtxt(f).reshape((Nd, Nd)).astype(int)
        return

    def save(self, path, solution):
        with open(path, "w") as f:
            numpy.savetxt(f, solution.values.reshape(Nd * Nd), fmt='%d')
        return

    def get_stat_of_solution(self):
        return self.best_data

    def solve(self):
        Nc = 1000
        Ne = int(0.05 * Nc)
        mutation_rate = 0.06

        self.population.seed(Nc, self.given)

        stale = 0
        generation = 0
        while(True):

            self.population.sort()

            if self.population.candidates[0].fitness == 1:
                print("Solution found at generation %d!" % generation)
                print(self.population.candidates[0].values)
                return self.population.candidates[0].values, generation

            if self.population.candidates[0].fitness != self.population.candidates[1].fitness:
                stale = 0
            else:
                stale += 1

            if stale >= 50:
                print("The population has gone stale. Re-seeding at generation %d!" % generation)
                self.population.seed(Nc, self.given)
                stale = 0
                mutation_rate = 0.06

            next_population = self.population.candidates[0:Ne]

            for count in range(Ne, Nc, 2):
                parent1 = self.population.candidates[random.randint(0, Nc - 1)].compete(self.population.candidates[random.randint(0, Nc - 1)])
                parent2 = self.population.candidates[random.randint(0, Nc - 1)].compete(self.population.candidates[random.randint(0, Nc - 1)])

                child1, child2 = parent1.crossover(parent2, crossover_rate=1.0)

                child1.mutate(mutation_rate, self.given)
                child2.mutate(mutation_rate, self.given)

                next_population.append(child1)
                next_population.append(child2)

            self.population.candidates = next_population
            self.population.update_fitness()

            mutation_rate = abs(numpy.random.normal(loc=0.0, scale=1, size=None))

            generation += 1


def check_solution(filename, solution):
    if os.path.exists("solutions/" + filename[0:-4] + "_s.txt"):
        with open("solutions/" + filename[0:-4] + "_s.txt", "r") as f:
            solution_orig = numpy.loadtxt(f).reshape((9, 9)).astype(int)
            print("Solution is " + str(numpy.array(solution == solution_orig).sum((0, 1)) == 81))


def run_test(filename):
    print(filename)
    s = Sudoku()
    s.load("./sudoku/" + filename)
    solution, generation = s.solve()
    check_solution(filename, solution)
    if solution is not None:
        return True, generation
    else:
        return False, generation


if __name__ == "__main__":
    if not os.path.exists("./Test10"):
        os.mkdir("./Test10")
    f = open("./Test10/" + "check.txt", "a")

    filenames = []
    if len(sys.argv) > 2:
        times_to_run_one_test = int(sys.argv[2])
        filenames.append(sys.argv[1])
    elif len(sys.argv) > 1:
        times_to_run_one_test = int(sys.argv[1])
        filenames = os.listdir("./sudoku")
    else:
        times_to_run_one_test = 5
        filenames = os.listdir("./sudoku")

    for filename in filenames:
        count_of_solved = 0
        generations = []
        ts = time.time()
        for i in range(times_to_run_one_test):
            print(str(i+1) + " launch")
            result_test, generation = run_test(filename)
            if result_test:
                count_of_solved += 1
                generations.append(generation)
        ts_finish = time.time()
        if count_of_solved:
            f.write(filename + " : generations = " + str(sum(generations) / len(generations)) + " time = " +
                    str((ts_finish - ts)/len(generations)) + " launchs = " + str(times_to_run_one_test)+'\n')
        else:
            f.write(filename + " : solution not sound\n")
        print("Time: ", ts_finish - ts)
    f.close()
