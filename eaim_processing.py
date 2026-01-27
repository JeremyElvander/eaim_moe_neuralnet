import numpy as np
from itertools import product
from functools import reduce
from math import gcd
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import lxml.html as lx
import re
import time
import datetime
import os
import sys

def linear_independent_compounds(max_val=10, tolerance=1e-16):
    """
    a = NH4+, 
    b = Na+, 
    c = SO42−, 
    d = NO3−, 
    e = Cl−
    """
    compositions=[]
    for c, d, e in product(range(max_val + 1), repeat=3):
        rhs = 2 * c + d + e

        for a in range(rhs+1):
            b = rhs-a
            if b > max_val or a > max_val:
                continue
            compositions.append(np.array([a, b, c, d, e]))

    compositions=np.array(compositions)

    U, S, Vt = np.linalg.svd(compositions, full_matrices=False)
    rank=np.sum(S>tolerance)

    return compositions[np.unique(np.round(Vt[:rank] @ compositions.T, decimals=12), axis=1).argmax(axis=1)]

def valid_compounds(max_val=10, step=1):
    """
    a = NH4+, 
    b = Na+, 
    c = SO42−, 
    d = NO3−, 
    e = Cl−
    """
    unique = set()
    for c, d, e in product(range(0, max_val + 1, step), repeat=3):
        rhs = 2 * c + d + e

        for a in range(rhs+1):
            b = rhs-a
            if (a == 0 and b == 0) or (c == 0 and d == 0 and e == 0):
                continue
            if b <= max_val and a <= max_val:
                vector = np.array([a, b, c, d, e])
                scalar = reduce(gcd, vector)
                reduced = tuple((vector // scalar).tolist()) if scalar > 0 else tuple(vector.tolist())
                unique.add(reduced)
    return np.array(sorted(unique))

def generate_input_data(compounds, temp_step = 0.5, rh_step = 0.05, model=4, limit=None):
    #263.15
    temps = np.linspace(263.15, 330, num=int((330 - 263.15) / temp_step) + 1)
    #0.6
    RHs = np.linspace(0.6, 0.998, num=int((0.998 - 0.6) / rh_step) + 1)
    num_comps = len(compounds)
    num_rows = len(temps) * len(RHs) * num_comps
    max_rows = num_rows if limit is None else min(limit, num_rows)
    data = np.empty((num_rows, 21), dtype=np.float64)

    idx = 0
    for temp, rh in product(temps, RHs):
        for comp in compounds:
            if idx >= max_rows:
                break
            a, b, c, d, e = comp.tolist()
            data[idx] = [temp,1, 1, 1, 1, rh, 0.0,
                   a, b, c, d, e, 0.0, 0.0, 0.0, 4, 4, 4, 4, 3, 0]
            idx += 1
            if idx%1000000 == 0:
                print(f"{idx / num_rows:.0%} complete")
        if idx >= max_rows:
            break
    df = pd.DataFrame(data[:idx])

    #Fixing data types
    int_cols = [1, 2, 3, 4, 15, 16, 17, 18, 19, 20]
    float_cols = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)

    return df


def generate_eaim_data(df, filepath, limit=None, max_workers=1):
    #Setup processing batches
    batch_size = 100
    results=[]
    rows = len(df) if limit is None else min(limit, len(df))
    batches = [df.iloc[i:min(i+batch_size, rows)].copy() for i in range(0, rows, batch_size)]
    batch_length = len(batches)

    #Time logging
    def log(msg):
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | {msg}")

    #Code to retrieve e-aim data at 100 row increments
    def process_batch(batch):
        # Convert to CSV string (no index, no header)
        ions = pd.DataFrame(batch.iloc[:,np.r_[0, 5, 7:12]])
        
        ions.rename(columns={0:'TEMP',5:'RH', 7:'NH4+', 8:'NA+', 9:'SO42-', 10:'NO3-', 11:'CL-'}, inplace=True)
        csv_str = batch.to_csv(index=False, header=False)

        # Replace \n with \r\n to comply with E-AIM format
        formatted = csv_str.rstrip().replace('\n', '\r\n')
        #generate a payload for the type of calculation we want in E-AIM
        payload = {'wwwUsageType': 'calculation', 'wwwInputMode': 'batch', 
                'Model': 'ModelIV', 'iCaseInorg': 1, 'ExcludeWater': 'y',
                'wwwOutputMode': 'column', 'nCompounds': 0, 'tf': formatted}

        #make a POST request to calculate values for training data
        r = requests.post('https://www.aim.env.uea.ac.uk/cgi-bin/eaim', data=payload, timeout=200)
        tree = lx.fromstring(r.text)
        
        # Extract the <pre> block containing E-AIM output
        pre_text = tree.xpath('//pre')[0].text_content()
        with open(filepath, 'a') as file:
            file.write(pre_text)
        return 'done'

    #Parallel processing for process_batch code to improve runtime performance
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(future_to_batch):
            try:
                result = future.result()
                results.append(result)
                count += 1
                log(f'Downloaded Batch Text {count}/{batch_length}')
            except Exception as e:
                print(f"Batch {future_to_batch[future]} failed: {e}")
    # return pd.concat(results, ignore_index=True)
    return 'done'

def parse_text_data(txt_files, max_workers=1):
    #Setup processing batches
    # batch_size = 100
    results=[]
    # rows = len(df) if limit is None else min(limit, len(df))
    # batches = [df.iloc[i:min(i+batch_size, rows)].copy() for i in range(0, rows, batch_size)]
    # batch_length = len(batches)


    #Time logging
    def log(msg):
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | {msg}")

    #Code to retrieve e-aim data at 100 row increments
    def process_batch(file):
        # Convert to CSV string (no index, no header)
        data = pd.DataFrame(columns=['TEMP', 'RH', 'NH4+',  'NA+', 'SO42-', 'NO3-', 'CL-', 'n_H2O_aq',  'n_H2O_g', 'p_H2O', 'p_HNO3', 'p_HCl', 'p_NH3',  'p_H2SO4',  'amm_nit',  'amm_chl',  'phase'])
        def read_text(key_1, data_1, key_2, data_2, key_3, data_3):
             # Extract values from Column Output (1)
            n_H2O_aq_idx = key_1.index('n_H2O(aq)')
            n_H2O_g_idx = key_1.index('n_H2O(g)')
            standard_idx = key_1.index('n_H2SO4(g)')
            err_idx = key_1.index('Err')
            temp_idx = key_1.index('T')
            rh_indx = key_1.index('RH')

            pp_lst = []
            for col in reversed(key_1):
                if 'name' in col:
                    pp_lst.insert(0, key_1.index(col))
                if 's01' in col:
                    break

            section_1_data = []
            section_2_data = []
            section_3_data = []
        
            for i in range(len(data_1)):
                row = data_1[i]
                tokens = row.split()
                try:
                    _ = float(tokens[err_idx-1])
                except Exception as e:
                    print(tokens[err_idx-1])
                    print('error')
                    continue
                idx_1 = int(tokens[0])
                n_H2O_aq = float(tokens[n_H2O_aq_idx-1])

                n_H2O_g = float(tokens[n_H2O_g_idx-1])
                token_length = len(tokens)
                amm_nit = None
                amm_chl = None
                if token_length > standard_idx and n_H2O_aq == 0:
                    phase = 1 #Solid
                    for j in pp_lst:
                        if j > token_length:
                            continue
                        if tokens[j-1] == 'NH4NO3':
                            amm_nit = np.inf
                        elif tokens[j-1] == 'NH4Cl':
                            amm_chl = np.inf
                elif token_length > standard_idx and n_H2O_aq != 0:
                    phase = 2 #Mix
                else:
                    phase = 3 #Liquid
                temp = float(tokens[temp_idx-1])
                rh = float(tokens[rh_indx-1])
                section_1_data.append((idx_1, n_H2O_aq, n_H2O_g, phase, temp, rh))

            # Extract values from Column Output (2)
            
                row = data_2[i]
                tokens = row.split()
                idx_2 = int(tokens[0])
                p_H2O_g = float(tokens[-5])
                p_HNO3 = float(tokens[-4])
                p_HCl = float(tokens[-3])
                p_NH3 = float(tokens[-2])
                p_H2SO4 = float(tokens[-1])
                if amm_nit == None:
                    amm_nit = p_HNO3 * p_NH3
                if amm_chl == None:
                    amm_chl = p_HCl * p_NH3
                
                section_2_data.append((idx_2, p_H2O_g, p_HNO3, p_HCl, p_NH3, p_H2SO4, amm_nit, amm_chl))

                row = data_3[i]
                tokens = row.split()
                idx_3 = int(tokens[0])
                if len(tokens) == 6:
                    nh4 = 0.0
                    na = float(tokens[1])
                    so42 = float(tokens[2])
                    no3 = float(tokens[3])
                    cl = float(tokens[4])
                else:
                    nh4 = float(tokens[1])
                    na = float(tokens[2])
                    so42 = float(tokens[3])
                    no3 = float(tokens[4])
                    cl = float(tokens[5])

                section_3_data.append((idx_3, nh4, na, so42, no3, cl))
            # Combine both sections by row index
            combined = []
            for (i1, aq, g, p, temp, rh), (i2, p_H2O, p_HNO3, p_HCl, p_NH3, p_H2SO4, amm_nit, amm_chl), (i2, nh4, na, so42, no3, cl) in zip(section_1_data, section_2_data, section_3_data):
                assert i1 == i2
                combined.append({
                    'TEMP': temp,
                    'RH':rh,
                    'NH4+':nh4,
                    'NA+':na,
                    'SO42-':so42,
                    'NO3-':no3,
                    'CL-':cl,
                    'n_H2O_aq': aq,
                    'n_H2O_g': g, #Is ice needed?
                    'p_H2O': p_H2O,
                    'p_HNO3': p_HNO3,
                    'p_HCl': p_HCl,
                    'p_NH3': p_NH3,
                    'p_H2SO4': p_H2SO4,
                    'amm_nit':amm_nit,
                    'amm_chl':amm_chl,
                    'phase':p
                })
            batch_df=pd.DataFrame(combined).reset_index(drop=True)
            return batch_df
        univ_count = 0
        #ions = pd.DataFrame(columns=['TEMP','RH','NH4+','NA+','SO42-','NO3-','CL-'])
        with open(file, 'r') as text:
            lines = text.readlines()
        data_rows_1 = []
        data_rows_2 = []
        data_rows_3 = []

        in_section_1 = False
        in_section_2 = False
        in_section_3 = False

        count_1 = 1
        count_2 = 2
        count_3 = 3   

        key_1 = None
        key_2 = None
        key_3 = None
        for line in lines:
            # Detect section starts
            if 'Column Output (1):' in line:
                in_section_1 = True
                in_section_2 = False
                in_section_3 = False

                count_1 = 0
                continue
            elif 'Column Output (2):' in line:
                in_section_1 = False
                in_section_2 = True
                in_section_3 = False

                count_2 = 0
                continue
            elif line.strip().startswith('Column Output (3):'):
                in_section_1 = False
                in_section_2 = False
                in_section_3 = True

                count_3 = 0
                continue
            
            if in_section_1 and line.strip().startswith('I'):
                key_1 = line.strip().split()
            elif in_section_2 and line.strip().startswith("I"):
                key_2 = line.strip().split()
            elif in_section_3 and line.strip().startswith("I"):
                key_3 = line.strip().split()

            if in_section_1 and re.match(r'^\s*\d+', line):
                data_rows_1.append(line.strip())
                count_1 += 1
            elif in_section_2 and re.match(r'^\s*\d+', line):
                data_rows_2.append(line.strip())
                count_2 += 1
            elif in_section_3 and re.match(r'^\s*\d+', line):
                data_rows_3.append(line.strip())
                count_3 += 1
            
            if count_1 == count_2 == count_3:
                try:
                    new_data = read_text(key_1, data_rows_1, key_2, data_rows_2, key_3, data_rows_3)
                    data = pd.concat([data.reset_index(drop=True), new_data.reset_index(drop=True)], ignore_index=True, axis=0)
                    log(f'Batch {univ_count}/9651')
                    univ_count += 1
                    count_1 = 1
                    count_2 = 2
                    count_3 = 3

                    data_rows_1 = []
                    data_rows_2 = []
                    data_rows_3 = []

                    key_1 = None
                    key_2 = None
                    key_3 = None
                except Exception as e:
                    log(f'Failed Batch {univ_count}/9651: {e}')
                    univ_count += 1
                    count_1 = 1
                    count_2 = 2
                    count_3 = 3

                    data_rows_1 = []
                    data_rows_2 = []
                    data_rows_3 = []

                    key_1 = None
                    key_2 = None
                    key_3 = None
                    continue
        return data

    #Parallel processing for process_batch code to improve runtime performance
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(txt_files)}
        for future in as_completed(future_to_batch):
            try:
                result = future.result()
                results.append(result)
                #count += 1
                #log(f'Downloaded Batch Text {count}/{batch_length}')
            except Exception as e:
                print(f"Batch {future_to_batch[future]} failed: {e}")
    return pd.concat(results, ignore_index=True)

def requery(data):
    #Takes data, returns modified dataframe where inf values are removed from e-aim
    data = data[data['TEMP']!=1.]
    copy = data.copy()

    amm_nit_temps = data[data['amm_nit']==np.inf].groupby('TEMP').first().reset_index()
    amm_chl_temps = data[data['amm_chl']==np.inf].groupby('TEMP').first().reset_index()
    unique_temps = pd.concat([amm_nit_temps, amm_chl_temps])
    num_rows = len(unique_temps)
    query = np.empty((num_rows, 21), dtype=np.float64)

    for idx in range(num_rows):
        row = unique_temps.iloc[idx,:]
        query[idx] = [row['TEMP'],1, 1, 1, 1, row['RH'], 0.0,
                   row['NH4+'], row['NA+'], row['SO42-'], row['NO3-'], row['CL-'], 0.0, 0.0, 0.0, 4, 4, 4, 4, 3, 0]
    
    df = pd.DataFrame(query)

    #Fixing data types
    int_cols = [1, 2, 3, 4, 15, 16, 17, 18, 19, 20]
    float_cols = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)

    solid_temp_pp = pd.DataFrame(columns=['TEMP', 'amm_nit', 'amm_chl'])
    batches = [df.iloc[0:100,:].to_csv(index=False, header=False), df.iloc[100:200,:].to_csv(index=False, header=False), df.iloc[200:204,:].to_csv(index=False, header=False)]

    for csv_str in batches:
        # Replace \n with \r\n to comply with E-AIM format
        formatted = csv_str.rstrip().replace('\n', '\r\n')
        #generate a payload for the type of calculation we want in E-AIM
        payload = {'wwwUsageType': 'calculation', 'wwwInputMode': 'batch', 
                'Model': 'ModelIV', 'iCaseInorg': 1, 'ExcludeWater': 'y',
                'wwwOutputMode': 'normal', 'nCompounds': 0, 'tf': formatted}
        
        #make a POST request to calculate values for training data
        r = requests.post('https://www.aim.env.uea.ac.uk/cgi-bin/eaim', data=payload, timeout=200)
        tree = lx.fromstring(r.text)
        # Extract the <pre> block containing E-AIM output
        pre_text = tree.xpath('//pre')[0].text_content()  
        lines = pre_text.splitlines()

        in_pre_section = False
        in_solid_section = False
        in_end = False
        temp = None
        amm_nit = None
        amm_chl = None
        for line in lines:
            if 'Problem no.' in line:
                in_pre_section = True
                in_solid_section = False
                in_end = False
                continue
            elif 'Pressure (atm)' in line:
                in_pre_section = False
                in_solid_section = True
                in_end = False
                continue
            elif 'There is no mixed' in line:
                in_pre_section = False
                in_solid_section = False
                in_end = True
                continue
            elements = line.split()
            if in_pre_section and elements and elements[0]=='T':
                temp = float(elements[2])
            elif in_solid_section:
                match = re.search(r"=\s*([0-9Ee\+\-\.]+),\s*([A-Za-z0-9]+)\s*=", line)
                if match:
                    pp = match.group(1)
                    chem = match.group(2)

                if chem == 'NH4Cl':
                    amm_chl = float(pp)
                elif chem == 'NH4NO3':
                    amm_nit = float(pp)
                else:
                    continue
            elif in_end:
                solid_temp_pp.loc[len(solid_temp_pp)] = [temp, amm_nit, amm_chl]
                temp = None
                amm_nit = None
                amm_chl = None
                in_pre_section = False
                in_solid_section = False
                in_end = False
    solid_temp_pp = (
        solid_temp_pp
        .groupby("TEMP", as_index=False)
        .agg({
            "amm_nit": "max",
            "amm_chl": "max"
        })
    )
    merged = copy.merge(solid_temp_pp, on='TEMP', how='left', suffixes=('', '_solids'))
    merged['amm_nit'] = np.where((merged['amm_nit']==np.inf) & (merged['amm_nit_solids'].notna()), merged['amm_nit_solids'], merged['amm_nit'])
    merged['amm_chl'] = np.where((merged['amm_chl']==np.inf)& (merged['amm_chl_solids'].notna()), merged['amm_chl_solids'], merged['amm_chl'])
    merged = merged.drop(columns=['amm_nit_solids', 'amm_chl_solids'])
    return solid_temp_pp
    return merged

def example_usage(limit=10, text_path='ex/path', parsed_paths = ['text/paths', 'more/text']):
    #Generate valid compounds
    vectors = valid_compounds(10)
    #Generate input data with temperature, RH combinations
    df = pd.DataFrame(generate_input_data(vectors, limit=limit))

    #Generate data and write to text file
    generate_eaim_data(df, text_path, limit=limit)

    #Parse text data
    eaim_data = parse_text_data(parsed_paths)
    #Requery values that may display as infinity in column output
    eaim_data_final = requery(eaim_data)
    eaim_data_final.to_parquet('eaim_training_new.parquet')

