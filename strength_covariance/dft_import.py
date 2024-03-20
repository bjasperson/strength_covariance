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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy import stats
from textwrap import wrap
from strength_covariance.linear_model import create_X_y
import statsmodels.api as sm


def create_pipe():
    pca = PCA()
    pipe = Pipeline(steps=[('scale',StandardScaler()),
                        ('pca',pca)])
    return pipe

def se_unit_convert(df):
    #J/m^2 to eV/angstrom^2
    return df*6.241509e+18*1.0E-20

def get_df_dft(path = "data/dft.csv"):
    df_dft = pd.read_csv(path)
    se_list = ['surface_energy_111_fcc', 'surface_energy_121_fcc',
               'surface_energy_100_fcc', 
               'unstable_stack_energy_fcc',
               'intr_stack_fault_energy_fcc']
    df_dft[se_list] = se_unit_convert(df_dft[se_list])
    return df_dft

def get_boxplot(df_clean, 
                dft_predicted_strength, 
                save_fig = True):
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

    lower = (dft_predicted_strength['mean'] - dft_predicted_strength['obs_ci_lower']).to_list()
    upper = (dft_predicted_strength['obs_ci_upper']-dft_predicted_strength['mean']).to_list()
    ax.errorbar(dft_predicted_strength['species'],
                dft_predicted_strength['mean'], 
                yerr = (lower,upper), fmt='x', markersize=10., alpha=1.0, color="r",
                label='\n'.join(wrap("Predicted strength using DFT indicator properties",20)),
                elinewidth=2.0,
                capsize = 4)
    
    # # remove error bar (https://swdg.io/2015/errorbar-legends/)
    # handles, labels = ax.get_legend_handles_labels()
    # new_handles = []
    # for h in handles:
    #     #only need to edit the errorbar legend entries
    #     if isinstance(h, container.ErrorbarContainer):
    #         new_handles.append(h[0])
    #     else:
    #         new_handles.append(h)

    # ax.legend(new_handles, labels, fontsize=8)

    if save_fig == True:
        fig.savefig(f"strength_covariance/model_ays/dft_w_pi.pdf", bbox_inches = 'tight')
    return 


def get_prop_boxplot(df_clean, prop, prop_label, dft_value, save_fig = True):
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
        fig.savefig(f"strength_covariance/data_ays/dft_props/dft_{prop}.pdf", bbox_inches = 'tight')
    return fig

def main():
    all_dft_properties = ['bulk_modulus_fcc',
                          'c44_fcc',
                         #'C11-C12',
                          'surface_energy_111_fcc',
                          'surface_energy_121_fcc',
                          'surface_energy_100_fcc',
                          'unstable_stack_energy_fcc',
                          'intr_stack_fault_energy_fcc',
                          'lattice_constant_fcc',
                          'relaxed_formation_potential_energy_fcc',
                          'vacancy_migration_energy_fcc'
                          ]
    
    # model_properties = ['vacancy_migration_energy_fcc',
    #                     'surface_energy_100_fcc',
    #                     'lattice_constant_fcc']
    model_properties = ['unstable_stack_energy_fcc', 
                        'intr_stack_fault_energy_fcc', 
                        'c44_fcc']
 
    df, readme = data_import(clean=True)
    label_dict = import_label_dict()
    print(f"number of points initially: {len(df)}")

    # remove jammed
    df = df[df["SF_jamming"]!="yes"]
    print(f"number of points after removing jammed: {len(df)}")


    X_df, y = create_X_y(df, model_properties)
    readme += f"{len(X_df.columns)} factors: {X_df.columns}\n"

    pipe = create_pipe()
    X = X_df[model_properties]
    pipe.fit(X)
    X_scaled = pipe.transform(X)
    X_scaled = sm.add_constant(X_scaled, prepend=False)
    res = sm.OLS(y,X_scaled).fit()
    print(res.summary())
    y_pred = res.predict(X_scaled)
    r2_adj = r2_score(y,y_pred)
    
    df_dft = get_df_dft() #also converts SEs and SFE units
    X_dft_scaled = pipe.transform(df_dft[X_df.columns])
    X_dft_scaled = sm.add_constant(X_dft_scaled, prepend=False)
    y_pred_dft = res.predict(X_dft_scaled)

    predictions = res.get_prediction(X_dft_scaled)
    df_pi = predictions.summary_frame(alpha=0.05)
    df_pi['species'] = df_dft['species'].tolist()
    print(df_pi)
    get_boxplot(df, df_pi)
    

    
    for prop in all_dft_properties:
        get_prop_boxplot(df, prop, label_dict[prop], df_dft)
    
    return


if __name__=="__main__":
    main()