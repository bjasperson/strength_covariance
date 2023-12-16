import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strength_covariance.model_selection import basic_outlier_removal, filter_param_list
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy import stats
from uncertainty_quantification import data_import, model_create, r2_adj_fun

def pred_vs_actual_plot(df, y_pred, r2_adj, title, filename):
    params_string = ""
    x = df['strength_MPa']
    p = sns.scatterplot(data=df, x=x,y=y_pred,hue='species')
    p.plot(np.linspace(min(x),max(x),50),
        np.linspace(min(x),max(x),50))
    p.set_xlabel("actual strength [MPa]")
    p.set_ylabel("predicted strength [MPa]")
    p.set_title(title,fontsize=10)
    p.text(0.95, 0.01, f"Adjusted R\N{SUPERSCRIPT TWO} = {r2_adj:.3f}",
        verticalalignment='bottom', horizontalalignment='right',
        transform=p.transAxes, fontsize=10)
    #fig = plt.figure()
    #fig.add_axes(p)
    plt.savefig(f"./strength_covariance/model_ays/{filename}.png",dpi=300)

def r2_plot(r2_list, r2_adj_list, filename):
    fig,ax = plt.subplots()
    ax.plot(range(1,len(r2_list)+1),r2_list,'bx',label = "r2",)
    ax.plot(range(1,len(r2_list)+1),r2_adj_list,'r.',label = "r2 adj")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    plt.savefig(f"./strength_covariance/model_ays/{filename}.png",dpi=300)


def y_pred_loo(pipe,X,y):
    y_pred = []
    for y_index in range(len(y)):
        available_indexes = [j for j in range(len(y)) if j != y_index]
        X_available = X.loc[available_indexes]
        y_available = y.loc[available_indexes]
        pipe.fit(X_available,y_available)
        y_pred.append(pipe.predict(X)[y_index])
    return y_pred


def main():
    df_clean = data_import()
  
    params_list = ['lattice_constant','bulk_modulus','c44','c11','c12',
            'cohesive_energy','thermal_expansion_coeff_fcc','surface_energy_100_fcc',
            'extr_stack_fault_energy','intr_stack_fault_energy','unstable_stack_energy',
            'unstable_twinning_energy','relaxed_formation_potential_energy_fcc',
            'unrelaxed_formation_potential_energy_fcc','relaxation_volume_fcc']

    params_list_full = filter_param_list(df_clean, params_list)

    X_df = df_clean[params_list_full]
    y = df_clean['strength_MPa']
    imput = KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)
    X_df = pd.DataFrame(imput.fit_transform(X_df), columns = imput.feature_names_in_)

    #model = linear_model.LinearRegression()
    model = linear_model.Ridge()
    pca = PCA()

    pipe = Pipeline(steps=[('scale',StandardScaler()),
                           ('pca',pca),
                           ('lr',model)])
    pipe = TransformedTargetRegressor(regressor = pipe,
                                           transformer = StandardScaler())


    if True: #all factor eval
        # ignore gb coeff until better populated
        params_list_full = [i for i in params_list_full if "gb_coeff" not in i]

        df_corr = df_clean[['strength_MPa']+params_list_full].corr(numeric_only=True).round(2)
        df_corr = abs(df_corr['strength_MPa']).sort_values(ascending=False).dropna()
        corr_list = df_corr.index.to_list()
        corr_list = corr_list[1:] # remove strength from list

        # leave out point you are predicting
        r2_list = []
        r2_adj_list = []
        
        for i in range(1,len(corr_list)):
            print(f"factor count {i}")
            X = X_df[corr_list[:i]]# df_clean[corr_list[:i]]
            print(f"X shape = {X.shape}, y shape = {y.shape}")
            #y_pred = y_pred_loo(pipe,X,y)
            pipe.fit(X,y)
            y_pred = pipe.predict(X)
            r2_list.append(r2_score(y, y_pred))
            k = len(X.columns)
            n = len(y)
            r2_adj_list.append(r2_adj_fun(r2_list[-1], n, k))

        title = f"Linear model (leave one out) using all factors"
        pred_vs_actual_plot(df_clean, y_pred, r2_adj_list[-1], title, "linear_all_factors")
        r2_plot(r2_list, r2_adj_list,"linear_r2_plot")
        print(f"corr_list = {corr_list}")
    
    if False: #3 factor model, exclude jamming
        # still need to fix imputer to be applied at start??? no if we are assuming that data doesn't exist by practioner
        df_clean = df_clean[df_clean['SF_jamming']!='yes'].reset_index()
        print(f"number of points w/o jamming: {len(df_clean)}")
        pipe = model_create(model_type = "ridge") # had to switch to ridge due to colinearity issue/blows up for all factors
        params_short = ['c44_fcc','extr_stack_fault_energy_fcc','unstable_stack_energy_fcc']
        X = df_clean[params_short]
        y_pred = y_pred_loo(pipe,X,y)
        title2 = f"Linear model (leave one out) w/o jammed:\nc44, eSFE, uSFE (all FCC)"
        r2_adj = r2_score(y,y_pred)
        pred_vs_actual_plot(df_clean, y_pred, r2_adj, title2, "linear_3factors")


    return

if __name__ == "__main__":
    main()