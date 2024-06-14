#import kim_query
import statsmodels.api as sm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

plt.rcParams['figure.dpi'] = 400
# properties list: https://openkim.org/properties

def property_details(prop, query, fields, colm_rename):
    
    if prop == 'grain_energy':
        query["property-id"] = 'tag:brunnels@noreply.openkim.org,2016-02-18:property/grain-boundary-symmetric-tilt-energy-relaxed-relation-cubic-crystal'
        # query["meta.runner.driver.short-id"] = "TD_410381120771_003"
        query["meta.runner.driver.shortcode"] = "TD_410381120771"
        fields["tilt-angle.source-value"] = 1
        fields["tilt-angle.source-unit"] = 1
        fields["tilt-axis.source-value"] = 1
        fields["relaxed-grain-boundary-energy.source-value"] = 1
        fields["relaxed-grain-boundary-energy.source-unit"] = 1
        fields["meta.runner.kimcode"] = 1
        fields["basis-atom-coordinates.source-value"] = 1
        colm_rename['tilt-angle.source-value'] = 'angle'
        colm_rename['tilt-angle.source-unit'] = 'angle_unit'
        colm_rename['tilt-axis.source-value'] = 'tilt_axis'
        colm_rename['basis-atom-coordinates.source-value'] = 'basis_atom_coords'
        colm_rename['relaxed-grain-boundary-energy.source-value'] = 'grain_energy'
        colm_rename['relaxed-grain-boundary-energy.source-unit'] = 'grain_energy_unit'
        colm_rename['meta.runner.kimcode'] = 'testname_grain_energy'
    
    if prop == 'lattice_const':
        query["property-id"] = "tag:staff@noreply.openkim.org,2014-04-15:property/structure-cubic-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_475411767977_007"
        query["meta.runner.driver.shortcode"] = "TD_475411767977"
        fields["a.source-value"] = 1
        colm_rename['a.source-value'] = 'lattice_constant'
        colm_rename['meta.runner.kimcode'] = 'testname_lattice_const'
        
    if prop == 'elastic_const':
        query["property-id"] = "tag:staff@noreply.openkim.org,2014-05-21:property/elastic-constants-isothermal-cubic-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_011862047401_006"
        query["meta.runner.driver.shortcode"] = "TD_011862047401"
        fields["c11.source-value"] = 1
        fields["c12.source-value"] = 1
        fields["c44.source-value"] = 1
        fields["c11.source-unit"] = 1
        colm_rename['c11.source-value'] = 'c11'
        colm_rename['c12.source-value'] = 'c12'
        colm_rename['c44.source-value'] = 'c44'
        colm_rename['c11.source-unit'] = 'elastic_const_unit'
        colm_rename['meta.runner.kimcode'] = 'testname_elastic_const'
        
    if prop == 'bulk_modulus':
        query["property-id"] = "tag:staff@noreply.openkim.org,2014-04-15:property/bulk-modulus-isothermal-cubic-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_011862047401_006"
        query["meta.runner.driver.shortcode"] = "TD_011862047401"
        fields["isothermal-bulk-modulus.source-value"] = 1
        fields["isothermal-bulk-modulus.source-unit"] = 1
        colm_rename['isothermal-bulk-modulus.source-value'] = 'bulk_modulus'
        colm_rename['isothermal-bulk-modulus.source-unit'] = 'bulk_modulus_unit'
        colm_rename['meta.runner.kimcode'] = 'testname_bulk_mod'
             
    if prop == 'cohesive_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2014-04-15:property/cohesive-potential-energy-cubic-crystal"
        # query["meta.runner.driver.short-id"] = "TD_475411767977_007"
        query["meta.runner.driver.shortcode"] = "TD_475411767977"
        fields["cohesive-potential-energy.source-value"] = 1
        fields["cohesive-potential-energy.source-unit"] = 1
        colm_rename['cohesive-potential-energy.source-value'] = 'cohesive_energy'
        colm_rename['meta.runner.kimcode'] = 'testname_cohesive_energy'
        colm_rename['cohesive-potential-energy.source-unit'] = 'cohesive_energy_unit'

    if prop == 'thermal_expansion_coeff':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-07-30:property/linear-thermal-expansion-coefficient-cubic-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_522633393614_001"
        query["meta.runner.driver.shortcode"] = "TD_522633393614"
        fields["linear-thermal-expansion-coefficient.source-value"] = 1
        fields["linear-thermal-expansion-coefficient.source-unit"] = 1
        colm_rename["linear-thermal-expansion-coefficient.source-value"] = 'thermal_expansion_coeff'
        colm_rename["linear-thermal-expansion-coefficient.source-unit"] = 'thermal_expansion_coeff_unit'
        colm_rename['meta.runner.kimcode'] = 'testname_thermal_expansion_coeff'

    if prop == 'surface_energy':
        #cubic crystal
        query["property-id"] = "tag:staff@noreply.openkim.org,2014-05-21:property/surface-energy-cubic-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_955413365818_004"
        query["meta.runner.driver.shortcode"] = "TD_955413365818"
        fields["surface-energy.source-value"] = 1
        fields["surface-energy.source-unit"] = 1
        fields["miller-indices.source-value"] = 1
        colm_rename["surface-energy.source-value"] = "surface_energy"
        colm_rename["surface-energy.source-unit"] = "surface_energy_unit"
        colm_rename["miller-indices.source-value"] = 'surface_energy_surface'
        colm_rename['meta.runner.kimcode'] = 'testname_surface_energy'

    if prop == 'ideal_surface_energy':
        #cubic crystal
        query["property-id"] = "tag:staff@noreply.openkim.org,2014-05-21:property/surface-energy-ideal-cubic-crystal"
        # query["meta.runner.driver.short-id"] = "TD_955413365818_004"
        query["meta.runner.driver.shortcode"] = "TD_955413365818"
        fields["ideal-surface-energy.source-value"] = 1
        fields["ideal-surface-energy.source-unit"] = 1
        fields["miller-indices.source-value"] = 1
        colm_rename["ideal-surface-energy.source-value"] = "ideal_surface_energy"
        colm_rename["ideal-surface-energy.source-unit"] = "ideal_surface_energy_unit"
        colm_rename["miller-indices.source-value"] = 'ideal_surface_energy_surface'
        colm_rename['meta.runner.kimcode'] = 'testname_ideal_surface_energy'

    if prop == 'extr_stack_fault_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-05-26:property/extrinsic-stacking-fault-relaxed-energy-fcc-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_228501831190_002"
        query["meta.runner.driver.shortcode"] = "TD_228501831190"
        fields["extrinsic-stacking-fault-energy.source-value"] = 1
        fields["extrinsic-stacking-fault-energy.source-unit"] = 1
        colm_rename["extrinsic-stacking-fault-energy.source-value"] = "extr_stack_fault_energy"
        colm_rename["extrinsic-stacking-fault-energy.source-unit"] = "extr_stack_fault_energy_unit"
        colm_rename['meta.runner.kimcode'] = 'testname_extr_stack_fault_energy'

    if prop == 'intr_stack_fault_energy':  
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-05-26:property/intrinsic-stacking-fault-relaxed-energy-fcc-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_228501831190_002"
        query["meta.runner.driver.shortcode"] = "TD_228501831190"
        fields["intrinsic-stacking-fault-energy.source-value"] = 1
        fields["intrinsic-stacking-fault-energy.source-unit"] = 1
        colm_rename["intrinsic-stacking-fault-energy.source-value"] = "intr_stack_fault_energy"
        colm_rename["intrinsic-stacking-fault-energy.source-unit"] = "intr_stack_fault_energy_unit"
        colm_rename['meta.runner.kimcode'] = 'testname_intr_stack_fault_energy'

    if prop == 'unstable_stack_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-05-26:property/unstable-stacking-fault-relaxed-energy-fcc-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_228501831190_002"
        query["meta.runner.driver.shortcode"] = "TD_228501831190"
        fields["unstable-stacking-energy.source-value"] = 1
        fields["unstable-stacking-energy.source-unit"] = 1
        fields["unstable-slip-fraction.source-value"] = 1
        colm_rename["unstable-stacking-energy.source-value"] = "unstable_stack_energy"
        colm_rename["unstable-stacking-energy.source-unit"] = "unstable_stack_energy_unit"
        colm_rename["unstable-slip-fraction.source-value"] = "unstable_stack_energy_slip_fraction"
        colm_rename['meta.runner.kimcode'] = 'testname_unstable_stack_energy'

    if prop == 'unstable_twinning_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-05-26:property/unstable-twinning-fault-relaxed-energy-fcc-crystal-npt"
        # query["meta.runner.driver.short-id"] = "TD_228501831190_002"
        query["meta.runner.driver.shortcode"] = "TD_228501831190"
        fields["unstable-twinning-energy.source-value"] = 1
        fields["unstable-twinning-energy.source-unit"] = 1
        fields["unstable-slip-fraction.source-value"] = 1
        colm_rename["unstable-twinning-energy.source-value"] = "unstable_twinning_energy"
        colm_rename["unstable-twinning-energy.source-unit"] = "unstable_twinning_energy_unit"
        colm_rename["unstable-slip-fraction.source-value"] = "unstable_twinning_energy_slip_fraction"
        colm_rename['meta.runner.kimcode'] = 'testname_unstable_twinning_energy'

    if prop == 'monovacancy_relaxed_formation_potential_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-07-28:property/monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt"
        query["meta.runner.driver.short-id"] = "TD_647413317626_001"
        #query["meta.runner.driver.shortcode"] = "TD_647413317626"
        fields["relaxed-formation-potential-energy.source-value"] = 1
        fields["relaxed-formation-potential-energy.source-unit"] = 1
        colm_rename["relaxed-formation-potential-energy.source-value"] = "relaxed_formation_potential_energy"
        colm_rename["relaxed-formation-potential-energy.source-unit"] = "relaxed_formation_potential_energy_unit"
        colm_rename['meta.runner.kimcode'] = 'testname_mono_relaxed_form_pot_energy'
    
    if prop == 'monovacancy_unrelaxed_formation_potential_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-07-28:property/monovacancy-neutral-unrelaxed-formation-potential-energy-crystal-npt"
        query["meta.runner.driver.short-id"] = "TD_647413317626_001"
        # query["meta.runner.driver.shortcode"] = "TD_647413317626"
        fields["unrelaxed-formation-potential-energy.source-value"] = 1
        fields["unrelaxed-formation-potential-energy.source-unit"] = 1
        colm_rename["unrelaxed-formation-potential-energy.source-value"] = "unrelaxed_formation_potential_energy"
        colm_rename["unrelaxed-formation-potential-energy.source-unit"] = "unrelaxed_formation_potential_energy_unit"
        colm_rename['meta.runner.kimcode'] = 'testname_mono_unrelaxed_form_pot_energy'

    if prop == 'monovacancy_vacancy_migration_energy':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-09-16:property/monovacancy-neutral-migration-energy-crystal-npt"
        query["meta.runner.driver.short-id"] = "TD_554849987965_001"
        # query["meta.runner.driver.shortcode"] = "TD_554849987965"
        fields["vacancy-migration-energy.source-value"] = 1
        fields["vacancy-migration-energy.source-unit"] = 1
        colm_rename["vacancy-migration-energy.source-value"] = "vacancy_migration_energy"
        colm_rename["vacancy-migration-energy.source-unit"] = "vacancy_migration_energy_unit"
        colm_rename['meta.runner.kimcode'] = 'testname_vacancy_migr_energy'


    if prop == 'monovacancy_relaxation_volume':
        query["property-id"] = "tag:staff@noreply.openkim.org,2015-07-28:property/monovacancy-neutral-relaxation-volume-crystal-npt"
        query["meta.runner.driver.short-id"] = "TD_647413317626_001"
        # query["meta.runner.driver.shortcode"] = "TD_647413317626"
        fields["relaxation-volume.source-value"] = 1
        fields["relaxation-volume.source-unit"] = 1
        colm_rename["relaxation-volume.source-value"] = "relaxation_volume"
        colm_rename["relaxation-volume.source-unit"] = "relaxation_volume_unit"
        colm_rename['meta.runner.kimcode'] = 'testname_vacancy_relax_vol'
    
    return query, fields, colm_rename


def get_tests(props):
    #generic version
    #props is list of properties, i.e. ['a','elastic_constant', ...]
    #plan is to pass whole list, then cycle thru
     
    df_list = []
    for current in props:
        query = {}
        query["meta.type"] = "tr"
        
        fields = {}
        fields["meta.subject.kimcode"] = 1 #model name
        fields["species.source-value"] = 1 #element
        fields["host-wyckoff-species.source-value"] = 1 #species for vacancy tests
        fields["short-name.source-value"] = 1 #crystal_type
        fields["host-short-name.source-value"] = 1 #crystal_type for vacancy tests
        fields["meta.runner.kimcode"] = 1
        
        colm_rename = {'short-name.source-value':'crystal_type',
                       'host-short-name.source-value':'crystal_type',
                       'species.source-value':'species',
                       'host-wyckoff-species.source-value':'species',
                       'meta.subject.kimcode':'model',
                       'meta.runner.driver.short-id':'model'}
        
        query, fields, colm_rename = property_details(current, 
                                                      query, 
                                                      fields, 
                                                      colm_rename)
        
        results = openkim_search(query, fields)
        
        if len(results) != 0:
            df = pd.DataFrame(results)
            df = df.dropna()
            df.rename(columns = colm_rename, inplace = True)
            df['species'] = [i[0] for i in df['species']]
            if 'stack' in current or 'twinning' in current:
                df['crystal_type'] = "fcc"
            else:
                df['crystal_type'] = [i[0] for i in df['crystal_type']]

            if current == 'surface_energy':
                df['surface_energy_surface'] = [str(i) for i in df['surface_energy_surface']]
                df1  = df[list(df.columns.drop(['surface_energy_surface','surface_energy']))]
                df1 = df1.drop_duplicates()
                df2 = df.pivot(index = ['crystal_type','species','model'],columns = 'surface_energy_surface',values='surface_energy').reset_index()
                df = df1.merge(df2, how='outer',on=['crystal_type','species','model'])
                rename = {'[1, 1, 0]':'surface_energy_110',
                          '[1, 1, 1]':'surface_energy_111',
                          '[1, 2, 1]':'surface_energy_121',
                          '[1, 0, 0]':'surface_energy_100'}
                df.rename(rename, inplace=True,axis=1)
            if current == 'ideal_surface_energy':
                df['ideal_surface_energy_surface'] = [str(i) for i in df['ideal_surface_energy_surface']]
                df1  = df[list(df.columns.drop(['ideal_surface_energy_surface','ideal_surface_energy']))]
                df1 = df1.drop_duplicates()
                df2 = df.pivot(index = ['crystal_type','species','model'],columns = 'ideal_surface_energy_surface',values='ideal_surface_energy').reset_index()
                df = df1.merge(df2, how='outer',on=['crystal_type','species','model'])
                rename = {'[1, 1, 0]':'ideal_surface_energy_110',
                          '[1, 1, 1]':'ideal_surface_energy_111',
                          '[1, 2, 1]':'ideal_surface_energy_121',
                          '[1, 0, 0]':'ideal_surface_energy_100'}
                df.rename(rename, inplace=True,axis=1)            
            df_list.append(df)
        else:
            print(f"no results for {current}, skipping")
        
    return df_list


def openkim_search(query, fields, database = 'data'):
    result = requests.post("https://query.openkim.org/api",
                           data = {'query':json.dumps(query),
                                   'fields':json.dumps(fields),
                                   'database':database,
                                   'limit':'0',
                                   'flat':'on'}).json()
    return result


def get_prop_df(props, df_ref):
    df_list = get_tests(props)
    df_list = flatten_crystal_type(df_list)
    df_merge = df_ref

    print('df_ref size:  ',df_ref.shape)
    print('-----')

    for i in range(len(df_list)):
        print('current property: ',props[i])
        print('initial merged shape: ',df_merge.shape)
        df_merge = pd.merge(df_merge, df_list[i], on=['model','species'], how='left')
        print('final merged shape: ',df_merge.shape)
        print('-----')

    columns = [i for i in df_merge.columns.to_list() if 'unit' in i]
    df_merge['units'] = df_merge[columns].to_dict(orient='records')
    df_merge = df_merge.drop(columns=columns)
    
    columns = [i for i in df_merge.columns.to_list() if 'testname' in i]
    df_merge['testnames'] = df_merge[columns].to_dict(orient='records')
    df_merge = df_merge.drop(columns=columns)

    return df_merge


def flatten_crystal_type(df_list):
    for i,df in enumerate(df_list):
        df = df.drop_duplicates()
        df = df.pivot(index=['species','model'],columns = ['crystal_type'])
        df.columns = [f'{i}_{j}' for i,j in df.columns]
        df = df.reset_index()
        df_list[i] = df
    return df_list



def get_all_IPs():
    models = []
    for type_in in ["mo","sm"]:
        query = {"type":type_in}
        fields = {"kimcode":1}
        results = openkim_search(query, fields, database = "obj")
        models.extend([i['kimcode'] for i in results])

    split_results = [i.split("__")[0] for i in models]
    code = [i.split("__")[1] for i in models]
    authors = [i.split("_")[-3] for i in split_results]
    year = [i.split("_")[-2] for i in split_results]
    species = [i.split("_")[-1] for i in split_results]
    type1 = [i.split("_")[0] for i in split_results]
    type2 = [i.split("_")[1] for i in split_results]
    df = pd.DataFrame({"original_model":models,
                        "authors":authors,
                        "year":year,
                        "model_species":species,
                        "type1":type1,
                        "type2":type2,
                        "kim_code":code})

    return df

def strength_ledger_import(file_location):
    df_ips = pd.read_csv(file_location)
    df_ips['type'] = [i.split("_")[2] for i in df_ips.ported_kimcode]
    df_ips['authors'] = [i.split("_")[3] for i in df_ips.ported_kimcode]
    df_ips['year'] = [i.split("_")[4] for i in df_ips.ported_kimcode]
    df_ips['model_species'] = [i.split("_")[5] for i in df_ips.ported_kimcode]
    df_ips['llnl_code'] = [i.split("__")[1] for i in df_ips.ported_kimcode]

    df_models = get_all_IPs()
    
    print(f"shape of llnl data: {df_ips.shape}")

    # merge for EAMs: filter on EAM_Dynamo
    df1 = pd.merge(df_ips, df_models[(df_models['type1'] == 'EAM') & (df_models['type2'] == 'Dynamo')],
                  how='inner',on=['authors','year','model_species'])

    # merge on SNAP: no duplicates known
    df2 = pd.merge(df_ips, df_models[(df_models['type1'] == 'SNAP')],
                  how='inner',on=['authors','year','model_species'])

    # merge for others, use llnl code
    df3 = pd.merge(df_ips, df_models,
                  how='inner',
                  left_on=['authors','year','model_species','llnl_code'],
                  right_on=['authors','year','model_species','kim_code'])
    
    df = pd.concat([df1,df2,df3])

    print(f"final shape: {df.shape}")


    df = df.rename({"original_model":"model",
                    "ported_kimcode":"llnl_ported_kimcode"},
                    axis=1)
    
    study_details_list = ['potential_type','file_sys_num','authors','year','model_species']
    df['study_details'] = df[study_details_list].values.tolist()
    df = df.drop(study_details_list,axis=1)
    return df