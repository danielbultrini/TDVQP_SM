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


def load_final_results():
    name1, name2 = 'Single', 'MD'
    ### Read simulation output files
    df1,df_coef1=read_files(100,1000,'results_helix/new/param/single_',ending='.json')
    df2,df_coef2=read_files(100,1000,'results_helix/new/shot_100/single_',ending='.json')
    df3,df_coef3=read_files(100,1000,'results_helix/new/shot_1000/single_',ending='.json')
    df4,df_coef4=read_files(100,1000,'results_helix/new/shot_10000/single_',ending='.json')
    df5,df_coef5=read_files(100,1000,'results_helix/new/shot_100000/contsingle_',ending='.jsonchk')


    udf1,udf_coef1=read_files(100,1000,'results_helix/new/unparam/single_',ending='.json')
    udf2,udf_coef2=read_files(100,1000,'results_helix/new/unparam_shot_100/single_',ending='.json')
    udf3,udf_coef3=read_files(100,1000,'results_helix/new/unparam_shot_1000/single_',ending='.json')
    udf4,udf_coef4=read_files(100,1000,'results_helix/new/unparam_shot_10000/single_',ending='.json')
    udf5,udf_coef5=read_files(100,1000,'results_helix/new/unparam_shot_100000/single_',ending='.jsonchk')


    mdf1,mdf_coef1=read_files(100,1000,'results_helix/new/md/md_',ending='.json')
    mdf2,mdf_coef2=read_files(100,1000,'results_helix/new/md100/md_',ending='.json')
    mdf3,mdf_coef3=read_files(100,1000,'results_helix/new/md1000/md_',ending='.json')
    mdf4,mdf_coef4=read_files(100,1000,'results_helix/new/md10000/md_',ending='.json')
    mdf5,mdf_coef5=read_files(100,1000,'results_helix/new/md100000/contmd_',ending='.jsonchk')
    # dfs = pd.read_feather('results_helix/normal_df.ft') ## Pickle for faster load times
    # dfscoef = pd.read_feather('results_helix/coef_df.ft')

    df1['shots'] = '$\infty$'
    df2['shots'] = '$10^2$'
    df3['shots'] = '$10^3$'
    df4['shots'] = '$10^4$'
    df5['shots'] = '$10^5$'
    df1['shotsn'] =  10**6
    df2['shotsn'] =  10**2
    df3['shotsn'] =  10**3
    df4['shotsn'] =  10**4
    df5['shotsn'] =  10**5

    df_coef1['shots'] =  '$\infty$'
    df_coef2['shots'] =  '$10^2$'
    df_coef3['shots'] =  '$10^3$'
    df_coef4['shots'] =  '$10^4$'
    df_coef5['shots'] =  '$10^5$'
    df_coef1['shotsn'] =  10**6
    df_coef2['shotsn'] =  10**2
    df_coef3['shotsn'] =  10**3
    df_coef4['shotsn'] =  10**4
    df_coef5['shotsn'] =  10**5

    df1['parameterized'] ='yes'
    df2['parameterized'] ='yes'
    df3['parameterized'] ='yes'
    df4['parameterized'] ='yes'
    df5['parameterized'] ='yes'
    df_coef1['parameterized'] ='yes'
    df_coef2['parameterized'] ='yes'
    df_coef3['parameterized'] ='yes'
    df_coef4['parameterized'] ='yes'
    df_coef5['parameterized'] ='yes'
    df1['simulation'] ='single'
    df2['simulation'] ='single'
    df3['simulation'] ='single'
    df4['simulation'] ='single'
    df5['simulation'] ='single'
    df_coef1['simulation'] ='single'
    df_coef2['simulation'] ='single'
    df_coef3['simulation'] ='single'
    df_coef4['simulation'] ='single'
    df_coef5['simulation'] ='single'


    udf1['shots'] =  '$\infty$'
    udf2['shots'] =  '$10^2$'
    udf3['shots'] =  '$10^3$'
    udf4['shots'] =  '$10^4$'
    udf5['shots'] =  '$10^5$'
    udf1['shotsn'] =  10**6
    udf2['shotsn'] =  10**2
    udf3['shotsn'] =  10**3
    udf4['shotsn'] =  10**4
    udf5['shotsn'] =  10**5

    udf_coef1['shots'] =  '$\infty$'
    udf_coef2['shots'] =  '$10^2$'
    udf_coef3['shots'] =  '$10^3$'
    udf_coef4['shots'] =  '$10^4$'
    udf_coef5['shots'] =  '$10^5$'
    udf_coef1['shotsn'] =  10**6
    udf_coef2['shotsn'] =  10**2
    udf_coef3['shotsn'] =  10**3
    udf_coef4['shotsn'] =  10**4
    udf_coef5['shotsn'] =  10**5

    udf1['parameterized'] = 'no'
    udf2['parameterized'] = 'no'
    udf3['parameterized'] = 'no'
    udf4['parameterized'] = 'no'
    udf5['parameterized'] = 'no'
    udf_coef1['parameterized'] = 'no'
    udf_coef2['parameterized'] = 'no'
    udf_coef3['parameterized'] = 'no'
    udf_coef4['parameterized'] = 'no'
    udf_coef5['parameterized'] = 'no'
    udf1['simulation'] ='single'
    udf2['simulation'] ='single'
    udf3['simulation'] ='single'
    udf4['simulation'] ='single'
    udf5['simulation'] ='single'
    udf_coef1['simulation'] ='single'
    udf_coef2['simulation'] ='single'
    udf_coef3['simulation'] ='single'
    udf_coef4['simulation'] ='single'
    udf_coef5['simulation'] ='single'

    mdf1['shots'] =  '$\infty$'
    mdf2['shots'] =  '$10^2$'
    mdf3['shots'] =  '$10^3$'
    mdf4['shots'] =  '$10^4$'
    mdf5['shots'] =  '$10^5$'
    mdf1['shotsn'] =  10**6
    mdf2['shotsn'] =  10**2
    mdf3['shotsn'] =  10**3
    mdf4['shotsn'] =  10**4
    mdf5['shotsn'] =  10**5

    mdf1['parameterized'] ='no'
    mdf2['parameterized'] ='no'
    mdf3['parameterized'] ='no'
    mdf4['parameterized'] ='no'
    mdf5['parameterized'] ='no'
    mdf_coef1['shots'] =  '$\infty$'
    mdf_coef2['shots'] =  '$10^2$'
    mdf_coef3['shots'] =  '$10^3$'
    mdf_coef4['shots'] =  '$10^4$'
    mdf_coef5['shots'] =  '$10^5$'
    mdf_coef1['shotsn'] =  10**6
    mdf_coef2['shotsn'] =  10**2
    mdf_coef3['shotsn'] =  10**3
    mdf_coef4['shotsn'] =  10**4
    mdf_coef5['shotsn'] =  10**5
    mdf_coef1['parameterized'] ='no'
    mdf_coef2['parameterized'] ='no'
    mdf_coef3['parameterized'] ='no'
    mdf_coef4['parameterized'] ='no'
    mdf_coef5['parameterized'] ='no'
    mdf1['simulation'] ='MD'
    mdf2['simulation'] ='MD'
    mdf3['simulation'] ='MD'
    mdf4['simulation'] ='MD'
    mdf5['simulation'] ='MD'
    mdf_coef1['simulation'] ='MD'
    mdf_coef2['simulation'] ='MD'
    mdf_coef3['simulation'] ='MD'
    mdf_coef4['simulation'] ='MD'
    mdf_coef5['simulation'] ='MD'

    dfs = pd.concat([df1,df2,df3,df4,df5,udf1,udf2,udf3,udf4,udf5,mdf1,mdf2,mdf3,mdf4,mdf5])
    dfscoef = pd.concat([df_coef1,df_coef2,df_coef3,df_coef4,df_coef5,udf_coef1,udf_coef2,udf_coef3,udf_coef4,udf_coef5
                        ,mdf_coef1,mdf_coef2,mdf_coef3,mdf_coef4,mdf_coef5])
    return dfs, dfscoef
