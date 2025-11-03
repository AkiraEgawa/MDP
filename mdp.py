import json as j

# In future iterations, I'll swap this out to ask for file name or use smth like argc argv* or smth
with open("gridworld.json", "r") as f:
    mdp = j.load(f)

# Gimme values
states = mdp["states"]
actions = mdp["actions"] # This in itself is a Json as well
terminal_states = set(mdp["terminal_states"])
gamma = mdp["gamma"]

# To iterate through the thingies

"""
Thinking to self for the creation of this function
So, we are given the Q and R for the actions (first version only)
We want to create V for each section, then call max or argmax to get the Q and R
V0 is always just 0, so make that first
V should always be same length as number of states
"""
def valueIt(mdp, theta=1e-6):
    V = {s: 0.0 for s in states}
    Iterations = {}
    itNum = 0
    while True:
        Iterations[itNum] = V.copy()
        itNum+=1
        delta = 0
        for s in states:
            if s in terminal_states:
                continue
            v = V[s]
            """
            Look for each action in the current state in this loop
            Calculate q values for each action
            Take max q value as new v
            """
            q_values = []
            for a in actions: # For each action, calculate the q values
                if s not in actions[a]:
                    continue
                q = 0
                for trans in actions[a][s]: # For each possible result of action sum up the q:
                    # add to q
                    q += trans["prob"]*(trans["reward"]+gamma*V[trans["next_state"]])
                q_values.append(q)
            V[s] = max(q_values) if q_values else 0
            delta = max(delta,abs(v-V[s])) # Delta is the largest jump of the optimal value at the current iteration

        if delta < theta:
            break
    
    # Iterations are now done, we have V*
    policy = {}
    for s in states:
        if s in terminal_states:
            policy[s] = None
            continue
        best_action = None
        best_value = float("-inf")
        for a in actions: # Every action you can take in the state
            if s not in actions[a]:
                continue
            # Use the q formula you created
            q=0
            for trans in actions[a][s]:
                q += trans["prob"]*(trans["reward"]+gamma*V[trans["next_state"]])
            # Now we have the reward possible from the action in s
            if q > best_value:
                best_value = q
                best_action = a
        policy[s] = best_action
    return V,policy,Iterations


print(valueIt(mdp)[1])