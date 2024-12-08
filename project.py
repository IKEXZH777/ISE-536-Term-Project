from tqdm import tqdm
import pandas as pd
from ortools.sat.python import cp_model


# Tested in python 3.10
# pip install pandas tqdm
# pip install ortools ## google's optimization package


# Container constraints
WEIGHT_CAP = 45000  
VOL_CAP = 3600   
PALLET_CAP = 60


class ProgressCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, total_bins, progress_bar, num_orders, checkpoint_file, x_vars):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.total_bins = total_bins
        self.progress_bar = progress_bar
        self.num_orders = num_orders
        self.checkpoint_file = checkpoint_file
        self.x_vars = x_vars
        self.container_assignments = [-1] * num_orders

    def on_solution_callback(self):
        # Update progress bar each time a solution is found
        self.progress_bar.update(1)
        current_bins = int(self.ObjectiveValue())
        print(f"Current Best: {current_bins} bins")

        # Save intermediate container assignments
        for i in range(self.num_orders):
            for j in range(self.total_bins):
                if self.Value(self.x_vars[i, j]):
                    self.container_assignments[i] = j

        # Save checkpoint to CSV
        checkpoint_df = pd.DataFrame({
            "Order": range(self.num_orders),
            "Container": self.container_assignments
        })
        checkpoint_df.to_csv(self.checkpoint_file, index=False)


def greedy_search(order_weights, order_volumes, order_pallets):
    """
    Greedy search method
    """
    num_orders = len(order_weights)
    remaining_weights = [WEIGHT_CAP]
    remaining_volumes = [VOL_CAP]
    remaining_pallets = [PALLET_CAP]
    container_assignments = [-1] * num_orders
    
    for i in range(num_orders):
        assigned = False
        for j in range(len(remaining_weights)):
            if (remaining_weights[j] >= order_weights[i] and 
                remaining_volumes[j] >= order_volumes[i] and 
                remaining_pallets[j] >= order_pallets[i]):
                # Assign order to this container
                remaining_weights[j] -= order_weights[i]
                remaining_volumes[j] -= order_volumes[i]
                remaining_pallets[j] -= order_pallets[i]
                container_assignments[i] = j
                assigned = True
                break

        # Create a new container
        if not assigned:    
            remaining_weights.append(WEIGHT_CAP - order_weights[i])
            remaining_volumes.append(VOL_CAP - order_volumes[i])
            remaining_pallets.append(PALLET_CAP - order_pallets[i])
            container_assignments[i] = len(remaining_weights) - 1

    result_df = pd.DataFrame({
        "Order": range(num_orders),
        "Container": container_assignments
    })
    output_file = './container_assignments_greedy.csv'
    result_df.to_csv(output_file, index=False)

    # # Solution sanity check
    # data = pd.read_csv(csv_file)
    # unique_containers = data['Container'].nunique()
    return container_assignments, len(remaining_weights)


def exact_search(order_weights, order_volumes, order_pallets, greedy_search_ret=None):
    order_weights = [int(w) for w in order_weights]
    order_volumes = [int(v) for v in order_volumes]
    order_pallets = [int(p) for p in order_pallets]

    num_orders = len(order_weights)
    if greedy_search_ret:
        max_bins = greedy_search_ret
    else:
        max_bins = num_orders  # Maximum possible containers (one order per container)

    # Create the CP-SAT model
    model = cp_model.CpModel()

    # Decision variables
    x = {}  # x[i][j] = 1 if order i is placed in bin j
    y = []  # y[j] = 1 if bin j is used
    for i in range(num_orders):
        for j in range(max_bins):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")
    for j in range(max_bins):
        y.append(model.NewBoolVar(f"y_{j}"))

    # Constraints
    if greedy_search_ret:
        model.Add(sum(y) <= greedy_search_ret)

    for i in range(num_orders):
        model.Add(sum(x[i, j] for j in range(max_bins)) == 1)
    for j in range(max_bins):
        model.Add(sum(order_weights[i] * x[i, j] for i in range(num_orders)) <= WEIGHT_CAP * y[j])
        model.Add(sum(order_volumes[i] * x[i, j] for i in range(num_orders)) <= VOL_CAP * y[j])
        model.Add(sum(order_pallets[i] * x[i, j] for i in range(num_orders)) <= PALLET_CAP * y[j])

    model.Minimize(sum(y))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 10 # Use multi-threads

    with tqdm(total=num_orders, desc="Solving Progress") as progress_bar:
        callback = ProgressCallback(max_bins, progress_bar, num_orders, './container_assignments_exact.csv', x)
        status = solver.SolveWithSolutionCallback(model, callback)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return callback.container_assignments, solver.ObjectiveValue()
    else:
        print("No feasible solution found!")
        return None, None


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset ...")
    file_path = '/Users/xeniawang/Desktop/ISE536/Term Project/Term project data 1b.csv'
    data = pd.read_csv(file_path)
    orders = data[['Weight (lbs)', 'Volume (in3)', 'Pallets']].dropna() # in total 690 rows
    # orders = data[['Weight (lbs)', 'Volume (in3)', 'Pallets']].dropna().head(100) # for debug
    print("Dataset loaded")

    # Decision variables
    order_weights = orders['Weight (lbs)'].values
    order_volumes = orders['Volume (in3)'].values
    order_pallets = orders['Pallets'].values

    _, total_containers = greedy_search(order_weights, order_volumes, order_pallets)
    print(f"Using greedy search (near-optimal solution), the total containers required is {total_containers}")

    print("Start solving the problem using exact search method ...")
    _, total_containers = exact_search(order_weights, order_volumes, order_pallets)
    print(f"Using exact search (optimal solution), the total containers required is {total_containers}")
