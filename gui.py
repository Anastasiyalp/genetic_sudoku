import pygame
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
            if p%100 == 0:
                print p
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

    def load(self, given):
        self.given = given
        return

    def save(self, path, solution):
        with open(path, "w") as f:
            numpy.savetxt(f, solution.values.reshape(Nd * Nd), fmt='%d')
        return

    def get_stat_of_solution(self):
        return self.best_data

    def solve(self):
        global grid
        Nc = 1000
        Ne = int(0.05 * Nc)
        mutation_rate = 0.06

        self.population.seed(Nc, self.given)

        stale = 0
        generation = 0
        while(True):
            self.population.sort()
            pygame.event.pump()
            screen.fill((255, 255, 255))
            draw()
            draw_box()
            pygame.display.update()
            pygame.time.delay(20)
            grid = self.population.candidates[0].values
            # white color background\
            screen.fill((255, 255, 255))

            draw()
            draw_box()
            pygame.display.update()
            pygame.time.delay(50)


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


pygame.font.init()

screen = pygame.display.set_mode((500, 600))
pygame.display.set_caption("SUDOKU SOLVER USING GENETIC ALGORITHM")
# img = pygame.image.load('icon.png')
# pygame.display.set_icon(img)

x = 0
y = 0
dif = 500 / 9
val = 0
grid = [[0, 5, 0, 0, 9, 0, 0, 0, 0],
        [0, 0, 4, 8, 0, 0, 0, 0, 9],
        [0, 0, 0, 1, 0, 7, 2, 8, 0],
        [5, 6, 0, 0, 0, 0, 1, 3, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 7, 3, 0, 0, 0, 0, 4, 2],
        [0, 2, 1, 5, 0, 8, 0, 0, 0],
        [6, 0, 0, 0, 0, 3, 8, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 6, 0]]
# grid = [
#     [7, 8, 0, 4, 0, 0, 1, 2, 0],
#     [6, 0, 0, 0, 7, 5, 0, 0, 9],
#     [0, 0, 0, 6, 0, 1, 0, 7, 8],
#     [0, 0, 7, 0, 4, 0, 2, 6, 0],
#     [0, 0, 1, 0, 5, 0, 9, 3, 0],
#     [9, 0, 4, 0, 6, 0, 0, 0, 5],
#     [0, 7, 0, 3, 0, 0, 0, 1, 2],
#     [1, 2, 0, 0, 0, 7, 4, 0, 0],
#     [0, 4, 9, 2, 0, 6, 0, 0, 7]
# ]

font1 = pygame.font.SysFont("comicsans", 40)
font2 = pygame.font.SysFont("comicsans", 20)


def get_cord(pos):
    global x
    x = pos[0] // dif
    global y
    y = pos[1] // dif


# Highlight the cell selected
def draw_box():
    for i in range(2):
        pygame.draw.line(screen, (255, 0, 0), (x * dif - 3, (y + i) * dif), (x * dif + dif + 3, (y + i) * dif), 7)
        pygame.draw.line(screen, (255, 0, 0), ((x + i) * dif, y * dif), ((x + i) * dif, y * dif + dif), 7)

    # Function to draw required lines for making Sudoku grid


def draw():
    # Draw the lines

    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                # Fill blue color in already numbered grid
                pygame.draw.rect(screen, (0, 153, 153), (i * dif, j * dif, dif + 1, dif + 1))

                # Fill gird with default numbers specified
                text1 = font1.render(str(grid[i][j]), 1, (0, 0, 0))
                screen.blit(text1, (i * dif + 15, j * dif + 15))
                # Draw lines horizontally and verticallyto form grid
    for i in range(10):
        if i % 3 == 0:
            thick = 7
        else:
            thick = 1
        pygame.draw.line(screen, (0, 0, 0), (0, i * dif), (500, i * dif), thick)
        pygame.draw.line(screen, (0, 0, 0), (i * dif, 0), (i * dif, 500), thick)

    # Fill value entered in cell


def draw_val(val):
    text1 = font1.render(str(val), 1, (0, 0, 0))
    screen.blit(text1, (x * dif + 15, y * dif + 15))


# Raise error when wrong value entered
def raise_error1():
    text1 = font1.render("WRONG !!!", 1, (0, 0, 0))
    screen.blit(text1, (20, 570))


def raise_error2():
    text1 = font1.render("Wrong !!! Not a valid Key", 1, (0, 0, 0))
    screen.blit(text1, (20, 570))


# Check if the value entered in board is valid
def valid(m, i, j, val):
    for it in range(9):
        if m[i][it] == val:
            return False
        if m[it][j] == val:
            return False
    it = i // 3
    jt = j // 3
    for i in range(it * 3, it * 3 + 3):
        for j in range(jt * 3, jt * 3 + 3):
            if m[i][j] == val:
                return False
    return True


# Solves the sudoku board using Backtracking Algorithm
def solve(grid, i, j):
    while grid[i][j] != 0:
        if i < 8:
            i += 1
        elif i == 8 and j < 8:
            i = 0
            j += 1
        elif i == 8 and j == 8:
            return True
    pygame.event.pump()
    for it in range(1, 10):
        if valid(grid, i, j, it) == True:
            grid[i][j] = it
            global x, y
            x = i
            y = j
            # white color background\
            screen.fill((255, 255, 255))
            draw()
            draw_box()
            pygame.display.update()
            pygame.time.delay(20)
            if solve(grid, i, j) == 1:
                return True
            else:
                grid[i][j] = 0
            # white color background\
            screen.fill((255, 255, 255))

            draw()
            draw_box()
            pygame.display.update()
            pygame.time.delay(50)
    return False


# Display instruction for the game
def instruction():
    text1 = font2.render("PRESS D TO RESET TO DEFAULT / R TO EMPTY", 1, (0, 0, 0))
    text2 = font2.render("ENTER VALUES AND PRESS ENTER TO VISUALIZE", 1, (0, 0, 0))
    screen.blit(text1, (20, 520))
    screen.blit(text2, (20, 540))


# Display options when solved
def result():
    text1 = font1.render("FINISHED PRESS R or D", 1, (0, 0, 0))
    screen.blit(text1, (20, 570))


run = True
flag1 = 0
flag2 = 0
rs = 0
error = 0
# The loop thats keep the window running
while run:

    # White color background
    screen.fill((255, 255, 255))
    # Loop through the events stored in event.get()
    for event in pygame.event.get():
        # Quit the game window
        if event.type == pygame.QUIT:
            run = False
            # Get the mouse postion to insert number
        if event.type == pygame.MOUSEBUTTONDOWN:
            flag1 = 1
            pos = pygame.mouse.get_pos()
            get_cord(pos)
            # Get the number to be inserted if key pressed
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x -= 1
                flag1 = 1
            if event.key == pygame.K_RIGHT:
                x += 1
                flag1 = 1
            if event.key == pygame.K_UP:
                y -= 1
                flag1 = 1
            if event.key == pygame.K_DOWN:
                y += 1
                flag1 = 1
            if event.key == pygame.K_1:
                val = 1
            if event.key == pygame.K_2:
                val = 2
            if event.key == pygame.K_3:
                val = 3
            if event.key == pygame.K_4:
                val = 4
            if event.key == pygame.K_5:
                val = 5
            if event.key == pygame.K_6:
                val = 6
            if event.key == pygame.K_7:
                val = 7
            if event.key == pygame.K_8:
                val = 8
            if event.key == pygame.K_9:
                val = 9
            if event.key == pygame.K_RETURN:
                flag2 = 1
                # If R pressed clear the sudoku board
            if event.key == pygame.K_r:
                rs = 0
                error = 0
                flag2 = 0
                grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                # If D is pressed reset the board to default
            if event.key == pygame.K_d:
                rs = 0
                error = 0
                flag2 = 0
                grid = [
                    [7, 8, 0, 4, 0, 0, 1, 2, 0],
                    [6, 0, 0, 0, 7, 5, 0, 0, 9],
                    [0, 0, 0, 6, 0, 1, 0, 7, 8],
                    [0, 0, 7, 0, 4, 0, 2, 6, 0],
                    [0, 0, 1, 0, 5, 0, 9, 3, 0],
                    [9, 0, 4, 0, 6, 0, 0, 0, 5],
                    [0, 7, 0, 3, 0, 0, 0, 1, 2],
                    [1, 2, 0, 0, 0, 7, 4, 0, 0],
                    [0, 4, 9, 2, 0, 6, 0, 0, 7]
                ]
    if flag2 == 1:
        # if solve(grid, 0, 0) == False:
        s = Sudoku()
        s.load(grid)
        solution, generation = s.solve()
        if solution is None:
            error = 1
        else:
            rs = 1
        flag2 = 0
    if val != 0:
        draw_val(val)
        # print(x)
        # print(y)
        if valid(grid, int(x), int(y), val) == True:
            grid[int(x)][int(y)] = val
            flag1 = 0
        else:
            grid[int(x)][int(y)] = 0
            raise_error2()
            # pygame.time.delay(2000)
        val = 0

    if error == 1:
        raise_error1()
        # pygame.time.delay(2000)
    if rs == 1:
        result()
    draw()
    if flag1 == 1:
        draw_box()
    instruction()

    # Update window
    pygame.display.update()

# Quit pygame window
pygame.quit()