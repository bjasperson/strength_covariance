import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ast import literal_eval

def factor_usage_plot(save_loc = './strength_covariance/model_ays'):
    df_cov = pd.read_csv("./strength_covariance/data_ays/corr_clean_all.csv",index_col=0)
    df = pd.read_csv("./data/models_w_props.csv",index_col=0)
    top_models = pd.read_csv("./strength_covariance/model_ays/kfold_lr_models.csv",index_col=0)
    top_models = top_models.sort_values("rmse_cv_score",ascending=False)

    factor_list = []
    for line in top_models['factors']:
        factor_list.extend([i for i in literal_eval(line)])
    factor_list = list(set(factor_list))
    factor_list.sort()

    factor_dict = {}
    for factor in factor_list: 
        if "lattice_constant" in factor:
            factor_dict[factor] = "lattice constant"
        if "c11" in factor or "c12" in factor or "c44" in factor or "bulk_modulus" in factor:
            factor_dict[factor] = "elastic const"
        if "surface_energy" in factor:
            factor_dict[factor] = "surface energy"
        if "cohesive_energy" in factor:
            factor_dict[factor] = "cohesive energy"
        if "stack_fault_energy" in factor or "unstable_twinning" in factor or "unstable_stack" in factor:
            factor_dict[factor] = "stack/twin fault energy"
        if "formation_potential" in factor:
            factor_dict[factor] = "vac. form. energies"
        if "relaxation_volume" in factor:
            factor_dict[factor] = "vac. form. vol."
        if "vacancy_migration_energy" in factor:
            factor_dict[factor] = "vac. migr. energies"
        if "thermal_expansion_coeff" in factor:
            factor_dict[factor] = "thermal exp coeff"
    
    # save off group labels
    df_factor_dict = pd.DataFrame([[i,factor_dict[i]] for i in factor_dict])
    df_factor_dict.to_csv("./strength_covariance/model_ays/factor_groups.csv")
    
    factor_table = []
    for line in top_models['factors']:
        factor_table.append([factor_dict[i] for i in literal_eval(line)])
    df_factors = pd.DataFrame(factor_table)
    rmse = abs(top_models['rmse_cv_score']).to_list()

    df_factors = df_factors.replace(factor_dict)
    factor_list = [factor_dict[i] for i in factor_list]
    factor_list = list(set(factor_list))
    factor_list.sort()

    heatmap_list = []
    N_lines = 40
    for line in range(N_lines):
        line_tally = []
        current_line = df_factors.iloc[line,:].tolist()
        for col in factor_list:
            if col in current_line:
                line_tally.extend([rmse[line]])
            else:
                line_tally.extend([0])
        heatmap_list.append(line_tally)
    df_factor_table = pd.DataFrame(heatmap_list,
                                columns = factor_list)
    df_factor_table['model ranking'] = [i+1 for i in range(N_lines)]
    df_factor_table = df_factor_table.set_index('model ranking')
    df_factor_table = df_factor_table.transpose()

    mask = df_factor_table == 0

    #cmap = sns.color_palette("mako", as_cmap=True)
    cmap = sns.cubehelix_palette(start=.2, rot=-.5, light=0.8, as_cmap=True, reverse=True)
    fig,ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df_factor_table, mask = mask, 
                cmap = cmap, square=True, linewidth=.5, linecolor="grey",
                cbar_kws = dict(location='top',label="RMSE", shrink= 0.35))
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    ax.tick_params("both",labelsize=8)
    ax.invert_yaxis()
    ax.set_ylabel("Property Grouping", style='italic',  horizontalalignment = "left", rotation=90, labelpad=10, y=0) #y=1.1,
    ax.set_xlabel("Model Performance Ranking", style='italic')
    fig.savefig(f"{save_loc}/factor_usage.pdf", bbox_inches = "tight")

    return

if __name__ == "__main__":
    factor_usage_plot("./figures/main")