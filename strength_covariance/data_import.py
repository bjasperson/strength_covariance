import pandas as pd
import openKimInterface


# def add_gb_coeff(df):
#     df_gb = pd.read_csv("./data/df_merge.csv")
#     df_gb = df_gb[['species','model','tilt_axis','coeff']]
#     tilt_axis_map = {'[0, 0, 1]':'001',
#                      '[1, 1, 0]':'110',
#                      '[1, 1, 1]':'111',
#                      '[1, 1, 2]':'112'}
#     df_gb.tilt_axis = [tilt_axis_map[i] for i in df_gb.tilt_axis]
#     df_gb = df_gb.pivot(index=['species','model'],columns = ['tilt_axis'])
#     df_gb.columns = [f'gb_{i}_{j}' for i,j in df_gb.columns]
#     df_gb = df_gb.reset_index()
    
#     df = pd.merge(df,df_gb,on=['species','model'],how='left')
#     return df


def main():

    file_location = './data/ip_list.csv'

    df_strength = openKimInterface.strength_ledger_import(file_location)
    df_strength[df_strength.isna().any(axis=1)]

    openkim_props = ['lattice_const', 'bulk_modulus', 'elastic_const', 
                    'cohesive_energy', 'thermal_expansion_coeff', 
                    'surface_energy', 'ideal_surface_energy',
                    'extr_stack_fault_energy','intr_stack_fault_energy',
                    'unstable_stack_energy','unstable_twinning_energy',
                    'monovacancy_relaxed_formation_potential_energy',
                    'monovacancy_unrelaxed_formation_potential_energy',
                    'monovacancy_vacancy_migration_energy',
                    'monovacancy_relaxation_volume'] 

    df_strength = openKimInterface.get_prop_df(openkim_props,
                                            df_strength)

    df_strength = df_strength[df_strength['strength_MPa'].notna()]
    # df_strength = add_gb_coeff(df_strength)
    df_strength.to_csv('./data/models_w_props_full.csv',index=False)
    df_strength = df_strength[df_strength['disqualified'] != True]
    df_strength.to_csv('./data/models_w_props.csv',index=False)

if __name__ == "__main__":
    main()