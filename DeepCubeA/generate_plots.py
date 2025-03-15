import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_all(data_lst, env_name):
    tables = []
    data_items = list(data_lst.items())
    for key, data in data_items:
        df = pd.DataFrame(data)
        df["method"] = key
        try:
            if env_name != "pyraminx":
                df["is_solved"] = df["solutions"].apply(lambda lst: int(any([all(env.cube == np.arange(0,len(env.cube))) for env in lst])))
            else:
                df["is_solved"] = df["solutions"].apply(lambda lst: int(any([all(env.pyramid == np.arange(0,len(env.pyramid))) for env in lst])))
        except:
            df["is_solved"] = 1.0
        if key == "astar":
            df["num_moves"] = df["solutions"].apply(lambda lst: len(lst))
        else:
            df["num_moves"] = df["solutions"].apply(lambda lst: len(lst)-1)
        table = df.groupby(["method", "scrambles"], as_index=False).agg(solve_rate=("is_solved", "mean"),
                                                                 num_moves=("num_moves", lambda x:  np.mean(x * df.loc[x.index, "is_solved"] ) ))
        
        tables.append(table.sort_values(by=["scrambles"]))
    assert len(tables) == 2, "Not data for grouped barchart for two methods"
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.bar(tables[0]["scrambles"]-0.2, tables[0]["solve_rate"], 0.4, label=data_items[0][0], color="red")
    plt.bar(tables[1]["scrambles"]+0.2, tables[1]["solve_rate"], 0.4, label=data_items[1][0], color="blue")
    plt.legend()
    plt.title("Average Solve Rate of GBFS vs. A-star")
    plt.savefig(f"./{data_items[0][0]}-vs-{data_items[1][0]}-{env_name}-avg_solve_rate.png")
    
    plt.clf()

    plt.bar(tables[0]["scrambles"]-0.2, tables[0]["num_moves"], 0.4, label=data_items[0][0], color="red")
    plt.bar(tables[1]["scrambles"]+0.2, tables[1]["num_moves"], 0.4, label=data_items[1][0], color="blue")
    plt.legend()
    plt.title("Average Solve Moves GBFS vs. A-star")
    plt.savefig(f"./{data_items[0][0]}-vs-{data_items[1][0]}-{env_name}-avg_moves.png")
    plt.clf()
    
    

if __name__ == "__main__":
    env_name = "pyraminx"
    data1 = pd.read_pickle(f"../DeepCubeA/data/{env_name}/train/data_0.pkl")
    data2 = pd.read_pickle(f"../DeepCubeA/results/{env_name}/results.pkl")
    data2["scrambles"] = data1["scrambles"]
    plot_all({"gbfs": data1, "astar": data2}, env_name)