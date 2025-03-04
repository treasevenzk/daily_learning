from ortools.sat.python import cp_model

'''
问题设定：
4名员工、3个班次(早班、中班、晚班)、一周7天班
约束条件：
每个班次每天必须有一名员工、每名员工每天最多值一个班、每名员工每周工作天数不超过5天、员工不能连续工作超过3天
'''

def solve_scheduling_problem():
    model = cp_model.CpModel()

    num_employees = 4
    num_shifts = 3
    num_days = 7

    shifts = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                shifts[(e, d, s)] = model.NewBoolVar(f'shift_e{e}_d{d}_s{s}')
    
    # 约束1： 每个班次每天必须分配一名员工
    for d in range(num_days):
        for s in range(num_shifts):
            model.Add(sum(shifts[(e, d, s)] for e in range(num_employees)) == 1)

    # 约束2：每名员工每天最多值一个班
    for e in range(num_employees):
        for d in range(num_days):
            model.Add(sum(shifts[(e, d, s)] for s in range(num_shifts)) <= 1)

    # 约束3：每名员工每周工作天数不超过5天
    for e in range(num_employees):
        model.Add(sum(shifts[(e, d, s)] for d in range(num_days) for s in range(num_shifts)) <= 5) 
    
    # 约束4：员工不能连续工作超过3天
    for e in range(num_employees):
        for d in range(num_days - 3):
            works = []
            for i in range(4):
                day_var = model.NewBoolVar(f'works_e{e}_d{d+i}')
                model.Add(sum(shifts[(e, d+i, s)] for s in range(num_shifts)) == day_var)
                works.append(day_var)
            model.Add(sum(works) <= 3)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("找到解决方案!")

        days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        shift_names = ['早班', '中班', '晚班']

        for d in range(num_days):
            print(f"\n{days[d]}")
            for s in range(num_shifts):
                for e in range(num_employees):
                    if solver.Value(shifts[(e, d, s)]) == 1:
                        print(f"   {shift_names[s]}: 员工 {e+1}")

        for e in range(num_employees):
            work_days = sum(solver.Value(shifts[(e, d, s)])
                            for d in range(num_days)
                            for s in range(num_shifts))
            print(f"\n员工 {e+1} 本周工作 {work_days} 天")
    else:
        print("没有找到解决方案!")
    
if __name__ == "__main__":
    solve_scheduling_problem()