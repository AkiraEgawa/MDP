import numpy as np
import json
# This will hold a function to generate a json for gridworld scenarios
# I'm too lazy to manually write the actions for literally each block in the grid accounting for noise, so a script will do it for me

"""
Grid: the gridworld grid in numpy array format (0 for normal square, positive or negative number for reward and punishment respectively)
    The grid is given in the form of rows ([[row0],[row1],[row2]...]), positive blocks are terminal states (you exit the moment you touch and collect reward)
dest: file to write into for the json
noise: chance that the move moves you sideways instead
gamma: depleting reward
returns: nothing (you just go and open the file yourself after)
"""
def genGridJS(grid, dest, noise=0, gamma=1):
    nrows,ncols = grid.shape
    states = []
    actions = ["up", "down", "left", "right"]
    terminal_states=[]
    action_dict = {a: {} for a in actions}

    # function to generate name of states
    def state_name(r, c):
        return f"s_{r}_{c}"
    
    # defines the terminal states
    for r in range(nrows):
        for c in range(ncols):
            s = state_name(r,c)
            states.append(s)
            if grid[r,c] > 0:
                terminal_states.append(s)
    
    # Define movement
    def move(r,c,dr,dc):
        nr,nc = r+dr, c+dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            return nr,nc
        else:
            return r, c # This is a wall
        
    # Movement directions for the move function
    directions = {
        "up": (-1,0),
        "down": (1,0),
        "left": (0,-1),
        "right": (0,1)
    }

    # For when noise comes into effect
    sideways = {
        "up": [("left", 0, -1), ("right", 0, 1)],
        "down": [("left", 0, -1), ("right", 0, 1)],
        "left": [("up", -1, 0), ("down", 1, 0)],
        "right": [("up", -1, 0), ("down", 1, 0)]
    }

    for a in actions:
        for r in range(nrows):
            for c in range(ncols):
                s = state_name(r,c)
                if s in terminal_states:
                    continue # no actions available you exit
                transitions = []

                # Intended direction
                dr,dc = directions[a]
                nr, nc = move(r, c, dr, dc)
                main_next = state_name(nr, nc)
                reward = grid[nr,nc]
                transitions.append({
                    "prob": 1 - 2*noise,
                    "next_state": main_next,
                    "reward": float(reward)
                })

                # Sideways
                for _, dr, dc in sideways[a]:
                    nr, nc = move(r,c,dr,dc)
                    next_s = state_name(nr,nc)
                    reward = grid[nr,nc]
                    transitions.append({
                        "prob": noise,
                        "next_state": next_s,
                        "reward": float(reward)
                    })

                # normalize probability in case my math messes up or somebody inputs an imaginary number or smth
                total_p = sum(t["prob"] for t in transitions) # This is normally 1 outside of float errors in computers
                for t in transitions:
                    t["prob"] /= total_p

                action_dict[a][s] = transitions

    mdp = {
        "states": states,
        "actions": action_dict,
        "terminal_states": terminal_states,
        "gamma": gamma
    }

    # Save the mdp to the file given
    with open(dest, "w") as f:
        json.dump(mdp, f, indent = 2)
    
    print(f"MDP JSON saved in {dest}")

grid = np.array([
    [0, 0, 1],
    [0, -1, 0]
])
genGridJS(grid, "gridworld.json", noise=0.1)