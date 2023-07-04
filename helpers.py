import json 
import pandas as pd
import numpy as np

def read_files(numfiles,max_times,basename,ending='json'):
    results = []
    dfs = []
    fin_files = 0
    for i in range(0, numfiles):
        try:
            file = basename+f"{i}"+ending
            #file = f"results/unknown/trycorrect{i}.json"
            data = json.load(open(file))
            if data['final_i']>=max_times:
                results.append(data)
                for j in ['QC','ideal','exact']:
                    datadf = pd.DataFrame(data[f'{j}_coefficients'])
                    datadf=datadf.reset_index()
                    datadf = datadf.melt(id_vars='index',var_name='Adiabatic state',value_name='Population')
                    datadf['Setup:'] = f'{j}'
                    datadf['sample'] = i
                    datadf['Time'] = datadf['index']*data['dt']
                    dfs.append(datadf)
                fin_files+=1
        except:
            pass
    numfiles=fin_files
    datadf = pd.concat(dfs,ignore_index=True)
    stuff = [
        "fidelity_to_ideal",
        "fidelity_to_exact",
        "ideal_forces_el",
        "ideal_forces_nuc",
        "ideal_tot_forces",
        "ideal_velocities",
        "ideal_positions",
        "ideal_energy_el",
        "ideal_energy_Tnuc",
        "ideal_energy_Vnuc",
        "exact_forces_el",
        "exact_forces_nuc",
        "exact_tot_forces",
        "exact_velocities",
        "exact_positions",
        "exact_energy_el",
        "exact_energy_Tnuc",
        "exact_energy_Vnuc",
        "QC_forces_el",
        "QC_forces_nuc",
        "QC_tot_forces",
        "QC_velocities",
        "QC_positions",
        "QC_energy_el",
        "QC_energy_Tnuc",
        "QC_energy_Vnuc",
        "force",
        "err_force",
        "energy",
        "err_energy",
        "init_F",
        "final_F",
        "err_init_F",
        "err_fin_F",
        "iter_number",
        "times",
    ]

    store = {i: [] for i in stuff}
    store["sample"] = []

    for i in range(numfiles):
        for j in stuff:
            store[j].extend(results[i][j][0:max_times])
            # print(len(store[j]))
        store["sample"].extend([i] * max_times)
    # print(len(store["sample"]))

    df = pd.DataFrame(store)
    df["diff_ideal_exact"] = np.abs(df["fidelity_to_exact"] - df["fidelity_to_ideal"])
    df['ideal_energy'] = df['ideal_energy_el']+df['ideal_energy_Vnuc']+df['ideal_energy_Tnuc']
    df['QC_energy'] = df['QC_energy_el']+df['QC_energy_Vnuc']+df['QC_energy_Tnuc']
    df['QC_energy_rel']= df['QC_energy']-df['ideal_energy']
    df['exact_energy'] = df['exact_energy_el']+df['exact_energy_Vnuc']+df['exact_energy_Tnuc']
    df['exact_energy_rel'] = df['exact_energy']-df['ideal_energy']
    df['ideal_energy_rel'] = df['ideal_energy']-df['ideal_energy'][0]
    coeff_names  ={'exact_coefficients':'Exact',
    'ideal_coefficients':'Ideal',
    'QC_coefficients': 'TDVQP'}

    store_coef = {coeff_names[i]: [] for i in coeff_names}
    store_coef['Time'] = []
    store_coef['State'] = []
    store_coef["Sample"] = []
    for i in range(numfiles):
        for j in coeff_names:
            for k, res in enumerate(results[i][j][0:max_times]):        
                store_coef[coeff_names[j]].extend(res[0:15]) # only really need highest 5 populations even in larger simulations with superpositions.
                if j == 'exact_coefficients':
                    store_coef['Time'].extend([results[i]['times'][k]]*15)
                    store_coef["Sample"].extend([i] * 15)
                    store_coef['State'].extend(list(range(15)))
                    
    df_coef = pd.DataFrame(store_coef).melt(id_vars=('Time','State','Sample'),value_name='Population',var_name='Type')
    del results
    del store
    del store_coef
    del dfs
    return df, df_coef
