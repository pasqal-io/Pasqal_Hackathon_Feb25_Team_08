import numpy as np
import pandas as pd

#TODO: 
# Edit charging/discharging prices so that they are not uniform and depend on time
def ev_cost(string, params):
    """

    DICTIONARY LAYOUT

    The string has lenght 2*N*T  
    The params dictionary must contain states in the following order:
    params is a dictionary that needs to contain the following keys:
    dt: the timestep of the EV charging problem
    P0: charging power of the vehicles, assuming uniformity
    ec: charging price 
    ed: charging price 
    lambda: the (large) costant that decouples the fourth state
    """

    delta_t = params["delta_t"]
    P0 = params["P0"]
    ec = params["ec"]
    ed = params["ed"]
    Lambda = params["Lambda"]

    ct = delta_t*P0*(ec+ed)
    dt = delta_t*P0*(ec-ed)

    config = np.array(list(string), dtype=int)

    cost = 0
    for i in range(len(config),2):
        cost += (ct+Lambda)*config[i]*config[i+1]-Lambda*config[i+1]+((dt-ct)/2-Lambda)*config[i+1]
    
    return cost

def average_cost(dict, params):
    df = pd.DataFrame(list(dict.items()), columns=['string', 'counts'])
    df['costs'] = df['string'].apply(lambda x: ev_cost(x, params))
    avg_cost = (df["costs"]*df["counts"]).sum()/ df["counts"].sum()

    return avg_cost

def constraint_function(params,target_energy):

    P0 = params["P0"]

    constraint = 0
    #here insert cost function
    if constraint >= target_energy:
        return 1
    else:
        return 0
    
def constraint_function_total(dict,params,target_energies):
    return
def final_cost(dict,params,constraint_weight,cost_weight):
    return 