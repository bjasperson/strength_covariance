import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container
import seaborn as sns
from strength_covariance.model_selection import basic_outlier_removal, filter_param_list, data_import
from strength_covariance.explore import import_label_dict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from textwrap import wrap
from strength_covariance.linear_model import create_X_y, y_pred_loo, linear_model_create
import statsmodels.api as sm


def se_unit_convert(df):
    #J/m^2 to eV/angstrom^2
    return df*6.241509e+18*1.0E-20

def get_df_dft(path = "data/dft.csv",
               add_c11_c12 = False):
    df_dft = pd.read_csv(path)
    se_list = ['surface_energy_111_fcc', 
               #'surface_energy_121_fcc',
               #'surface_energy_100_fcc', 
               #'unstable_stack_energy_fcc',
               'intr_stack_fault_energy_fcc']
    df_dft[se_list] = se_unit_convert(df_dft[se_list])
    if add_c11_c12 == True:
        df_dft = add_elastic_consts(df_dft)
    return df_dft

def add_elastic_consts(df):
    df['c12_fcc'] = df['bulk_modulus_fcc'] - df['C11-C12']/3
    df['c11_fcc'] = df['c12_fcc'] + df['C11-C12']
    return df

def get_df_dft_prediction(df_pi):
    lower = (df_pi['mean'] - df_pi['obs_ci_lower']).to_list()
    upper = (df_pi['obs_ci_upper']-df_pi['mean']).to_list()
    species = df_pi['species'].to_list()
    mean = df_pi['mean'].to_list()
    df_pred = pd.DataFrame({"lower":lower,
                            "upper":upper,
                            "species":species,
                            "mean":mean})

    return df_pred

def get_boxplot(df_clean, 
                df_pred, 
                save_fig = True,
                save_loc = 'strength_covariance/model_ays',
                file_name = 'dft_w_pi',
                plot_errorbar = True):
    order_list = ["Ag","Al","Au","Cu","Ni","Pd","Pt"]
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(data = df_clean, 
                x="species", 
                y="strength_MPa", 
                order=order_list, 
                color = "0.8", 
                linewidth=1.0,
                fliersize=5.0,
                whis=0,
                flierprops={"marker":"."})
    ax.set_ylabel("Strength [MPa]")

    # add dft Xs
    ax.scatter(df_pred['species'],
               df_pred['mean'], 
               marker='x', 
               s=100., 
               alpha=1.0, 
               color="r",
               label='\n'.join(wrap("Predicted strength using DFT indicator properties",20)),)

    # add errorbars if desired
    if plot_errorbar == True:
        ax.errorbar(df_pred['species'],
                    df_pred['mean'], 
                    yerr = (df_pred['lower'],df_pred['upper']), fmt='.', markersize=0.0001, alpha=1.0, color="r",
                    #label='\n'.join(wrap("Predicted strength using DFT indicator properties",20)),
                    elinewidth=2.0,
                    capsize = 4)

    # ax.legend(new_handles, labels, fontsize=8)

    if save_fig == True:
        fig.savefig(f"{save_loc}/{file_name}.pdf", bbox_inches = 'tight')
    return 


def get_prop_boxplot(df_clean, 
                     prop, 
                     prop_label, 
                     dft_value, 
                     save_fig = True,
                     save_loc = "strength_covariance/data_ays/dft_props"):
    order_list = ["Ag","Al","Au","Cu","Ni","Pd","Pt"]
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4,3))
    g = sns.boxplot(data = df_clean, 
                x="species", 
                y=prop, 
                order=order_list, 
                color = "0.8", 
                linewidth=1.0,
                fliersize=5.0,
                whis=0,
                flierprops={"marker":"."})
    g.set_ylabel(prop_label)
    g.scatter(data = dft_value,
              x = "species",
              y = prop,
              marker = 'x',
              s=50.,
              color = "r")

    if save_fig == True:
        fig.savefig(f"{save_loc}/dft_{prop}.pdf", bbox_inches = 'tight')
    return fig


def error_plots(y, 
                y_pred,
                save_loc = "figures/dft"):

    error = y - y_pred
    rel_error = (y - y_pred)/y_pred
    
    fig,ax = plt.subplots(figsize=(3,3))
    ax.hist(error)
    ax.set_ylabel(r'$y-y_{pred}$')
    fig.savefig(f"{save_loc}/error.pdf", bbox_inches = "tight")
    
    fig,ax = plt.subplots(figsize=(3,3))
    ax.hist(rel_error)
    ax.set_ylabel(r'$(y-y_{pred})/y_{pred}$')
    fig.savefig(f"{save_loc}/rel_error.pdf", bbox_inches = "tight")


    

    return 


def main():
    all_dft_properties = [#'bulk_modulus_fcc',
                          'c44_fcc',
                          #'C11-C12',
                          #'c11_fcc',
                          #'c12_fcc',
                          'surface_energy_111_fcc',
                          #'surface_energy_121_fcc',
                          #'surface_energy_100_fcc',
                          #'unstable_stack_energy_fcc',
                          'intr_stack_fault_energy_fcc',
                          'lattice_constant_fcc',
                          'relaxed_formation_potential_energy_fcc',
                          'vacancy_migration_energy_fcc'
                          ]

    model_properties = ['vacancy_migration_energy_fcc',
                        'surface_energy_111_fcc',
                        'lattice_constant_fcc']
 
    df, readme = data_import(clean=True)
    label_dict = import_label_dict()
    print(f"number of points initially: {len(df)}")

    # remove jammed
    df = df[df["SF_jamming"]!="yes"].reset_index()
    print(f"number of points after removing jammed: {len(df)}")



    
    if False:
        X_dft_scaled = sm.add_constant(X_dft_scaled, prepend=False)

        # get predictions and prediction interval
        X_scaled = sm.add_constant(X_scaled, prepend=False)
        res = sm.OLS(y,X_scaled).fit()
        print(res.summary())
        y_pred = res.predict(X_scaled)
        r2_adj = r2_score(y,y_pred)
        y_pred_dft = res.predict(X_dft_scaled)
        predictions = res.get_prediction(X_dft_scaled)
        df_pi = predictions.summary_frame(alpha=0.05)
        df_pi['species'] = df_dft['species'].tolist()
        print(df_pi)
        df_pred = get_df_dft_prediction(df_pi)
        print(df_pred)

        get_boxplot(df, 
                    df_pred,
                    save_loc = "figures/main/dft")

    if True:
        X_df, y = create_X_y(df, model_properties)
        readme += f"{len(X_df.columns)} factors: {X_df.columns}\n"
        X = X_df[model_properties]

        # first, get estimate for RMSE
        pipe = linear_model_create("lr", add_imputer = True)
        y_pred = y_pred_loo(pipe, X, y)
        error_plots(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared = False)

        readme += "\n\n---------------------\ndft results\n\n"
        readme += f"factors used: {model_properties}\n\n"
        readme += f"rmse = {rmse}\n"
        readme += f"relative rmse = {rmse/np.mean(y)}\n"
        readme += f"2*rmse = {2*rmse}\n"  

        # now, prepare model for DFT 
        pipe = linear_model_create("lr", add_imputer = True)
        
        df_dft = get_df_dft() #also converts SEs and SFE units
        X_dft = df_dft[X_df.columns]     

        pipe.fit(X,y)
        y_pred_dft = pipe.predict(X_dft)
        lower = [2*rmse for i in range(len(df_dft))]
        upper = lower
        df_pred = pd.DataFrame({'mean':y_pred_dft,
                                'species':df_dft.species,
                                'lower':lower,
                                'upper':upper})
        print(df_pred)
        readme += df_pred.to_string()
        get_boxplot(df, 
                    df_pred,
                    save_loc = "figures/main",
                    file_name='dft_w_pi_LOO',
                    plot_errorbar=False)
        
        readme += "\n\n---------------------\nleave-one-out results\n\n"
        readme += pd.DataFrame({"strength":y,
                               "predicted":y_pred,
                               "error":(y - y_pred),
                               "rel error":(y - y_pred)/y_pred,
                               "species":df.species}
                               ).sort_values(['strength','species']).to_string()
        

        with open(f"./strength_covariance/model_ays/dft_readme.txt","w") as out:
             for line in readme:
                 out.write(line)
    
    if True:
        for prop in all_dft_properties:
            get_prop_boxplot(df, 
                            prop, 
                            label_dict[prop], 
                            df_dft,
                            save_loc = "figures/si")
    
    return

if __name__=="__main__":
    main()