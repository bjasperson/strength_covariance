import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strength_covariance.model_selection import basic_outlier_removal, filter_param_list, data_import
from explore import import_label_dict
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
from linear_model import create_X_y
import statsmodels.api as sm


def create_pipe():
    pca = PCA()
    pipe = Pipeline(steps=[('scale',StandardScaler()),
                        ('pca',pca)])
    return pipe

def se_unit_convert(df):
    #J/m^2 to eV/angstrom^2
    return df*6.241509e+18*1.0E-20

def get_df_dft():
    df_dft = pd.read_csv("data/dft.csv")
    se_list = ['surface_energy_111_fcc', 'surface_energy_112_fcc',
               'surface_energy_100_fcc', 'unstable_stack_energy_fcc',
               'intr_stack_fault_energy_fcc']
    df_dft[se_list] = se_unit_convert(df_dft[se_list])
    return df_dft

def get_boxplot(df_clean, dft_predicted_strength):
    order_list = ["Ag","Al","Au","Cu","Ni","Pd","Pt"]
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(data = df_clean, x="species", y="strength_MPa", order=order_list, color = "0.8", linewidth=0.5)
    ax.set_ylabel("Strength [MPa]")
    # ax.scatter(data = dft_predicted_strength, 
    #         x="species",
    #         y="mean", 
    #         marker=".", 
    #         color="b", 
    #         s=40, 
    #         label='\n'.join(wrap("Predicted Strength using DFT Indicator Properties",20))
    #         )
    lower = (dft_predicted_strength['mean'] - dft_predicted_strength['obs_ci_lower']).to_list()
    upper = (dft_predicted_strength['obs_ci_upper']-dft_predicted_strength['mean']).to_list()
    ax.errorbar(dft_predicted_strength['species'],
                dft_predicted_strength['mean'], 
                yerr = (lower,upper), fmt='.', markersize=10, alpha=1, color="b",
                label='\n'.join(wrap("Predicted Strength using DFT Indicator Properties",20)),
                elinewidth=3.0)
    ax.legend(fontsize=8)
    fig.savefig(f"strength_covariance/model_ays/dft_w_pi.pdf", bbox_inches = 'tight')

def main():
    params_list_full = ['bulk_modulus_fcc',
                        'c44_fcc',
                        'surface_energy_111_fcc',
                        'unstable_stack_energy_fcc',
                        'intr_stack_fault_energy_fcc'
                        ]  # for dft
 
    df, readme = data_import(clean=True)
    label_dict = import_label_dict()
    print(f"number of points initially: {len(df)}")

    # remove jammed
    df = df[df["SF_jamming"]!="yes"]
    print(f"number of points after removing jammed: {len(df)}")

    X_df, y = create_X_y(df, params_list_full)
    readme += f"{len(X_df.columns)} factors: {X_df.columns}\n"

    pipe = create_pipe()
    X = X_df[params_list_full]
    pipe.fit(X)
    X_scaled = pipe.transform(X)
    X_scaled = sm.add_constant(X_scaled, prepend=False)
    res = sm.OLS(y,X_scaled).fit()
    print(res.summary())
    y_pred = res.predict(X_scaled)
    r2_adj = r2_score(y,y_pred)
    
    df_dft = get_df_dft()
    X_dft_scaled = pipe.transform(df_dft[X_df.columns])
    X_dft_scaled = sm.add_constant(X_dft_scaled, prepend=False)
    y_pred_dft = res.predict(X_dft_scaled)

    predictions = res.get_prediction(X_dft_scaled)
    df_pi = predictions.summary_frame(alpha=0.05)
    df_pi['species'] = df_dft['species'].tolist()
    print(df_pi)
    get_boxplot(df, df_pi)
    return


if __name__=="__main__":
    main()