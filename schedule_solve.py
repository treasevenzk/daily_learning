from ortools.sat.python import cp_model

def solve_sudoku():
    model = cp_model.CpModel()

    cells = {}
    for i in range(9):
        for j in range(9):
            cells[i, j] = model.NewIntVar(1, 9, f'cell{i}{j}')
        
    print(cells)

    for j in range(9):
        model.AddAllDifferent(cells[i, j] for j in range(9))

    for j in range(9):
        model.AddAllDifferent(cells[i, j] for i in range(9))

    for box_i in range(3):
        for box_j in range(3):
            box_vars = []
            for i in range(3):
                for j in range(3):
                    box_vars.append(cells[3 * box_i + i, 3 * box_j + j])
            model.AddAllDifferent(box_vars)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        for i in range(9):
            line = ""
            for j in range(9):
                line += str(solver.Value(cells[i, j])) + " "
            print(line)

solve_sudoku()