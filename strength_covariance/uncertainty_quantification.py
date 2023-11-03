import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_selection import basic_outlier_removal, filter_param_list
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy import stats




def uncert_bootstrap(X,y,pipe,n_iter,readme):

    y_pred_data = []
    y_pred_mean = []
    y_pred_lower = []
    y_pred_upper = []
    ci_range = 95.0
    readme.append(f"CI range: {ci_range}\n")
    for i in range(len(y)):
        print(f"data point {i} of {len(y)}")
        available_indexes = [j for j in range(len(y)) if j != i]
        X_available = X.loc[available_indexes]
        y_available = y.loc[available_indexes]
        y_pred_bootstrap = []

        for j in range(n_iter):
            X_train = resample(X_available)
            in_indexes = X_train.index.to_list()
            y_train = y_available.loc[in_indexes]
            pipe.fit(X_train, y_train)
            y_pred_value = pipe.predict(X)[i]
            y_pred_bootstrap.append(y_pred_value)
        
        y_pred_data.append(y_pred_bootstrap)
        lower, upper = stats.scoreatpercentile(y_pred_bootstrap,[2.5,97.5])
        mean = np.mean(y_pred_bootstrap)
        y_pred_lower.append(mean-lower)
        y_pred_upper.append(upper-mean)
        y_pred_mean.append(mean)
    
    y_pred = {'y_pred_mean':y_pred_mean,
              'y_pred_lower':y_pred_lower,
              'y_pred_upper':y_pred_upper}

    return y_pred, readme


def r2_adj_fun(r2,n,k):
    return (1 - ((1-r2)*(n-1)/(n-k-1)))


def boostrap_plot(df, y_pred, r2_adj, filename, params, title=True):
    params_string = ""
    for param in params:
        params_string += (param + ", ")
    params_string = params_string[:-2]
    y_pred_lower = y_pred['y_pred_lower']
    y_pred_upper = y_pred['y_pred_upper']
    y_pred_mean = y_pred['y_pred_mean']
    y = df['strength_MPa']
    y_pred_limits = np.array([y_pred_lower,y_pred_upper])
    p = sns.scatterplot(data=df, x='strength_MPa',y='strength_pred',hue='species')
    p.errorbar(y,y_pred_mean, yerr=y_pred_limits, fmt='.',markersize=0.001, alpha=0.75)
    p.plot(np.linspace(min(y),max(y),50),
           np.linspace(min(y),max(y),50))
    p.set_xlabel("actual strength [MPa]")
    p.set_ylabel("predicted strength [MPa]")
    if title==True:
        p.set_title(f"Bootstrap uncertainty results, 95% CI (point withheld)\n{params_string}\nAdjusted r2 using mean = {r2_adj:.3f}",fontsize=10)
    p.text(0.95, 0.01, f"Adjusted R\N{SUPERSCRIPT TWO} using mean = {r2_adj:.3f}",
           verticalalignment='bottom', horizontalalignment='right',
           transform=p.transAxes, fontsize=10)
    #fig = plt.figure()
    #fig.add_axes(p)
    plt.savefig(f"./strength_covariance/model_ays/{filename}.png",dpi=300)


def perform_bootstrap(df, params_list_in, pipe, n_bootstrap, filename, title=True):
    readme = []
    readme.append(f"factor list: {params_list_in}\n")
    X = df[params_list_in]
    y = df['strength_MPa']

    y_pred, readme = uncert_bootstrap(X,y,pipe,n_bootstrap,readme)

    r2 = r2_score(y, y_pred['y_pred_mean'])
    k = len(X.columns)
    n = len(y)
    r2_adj = r2_adj_fun(r2, n, k)

    df['strength_pred'] = y_pred['y_pred_mean']

    boostrap_plot(df, y_pred, r2_adj, filename, params_list_in, title=title)

    with open(f"./strength_covariance/model_ays/{filename}_readme.txt", "w") as text_file:
        for line in readme:
            text_file.write(line)
    
    df = pd.concat([df,pd.DataFrame(y_pred)], axis=1)
    df.to_csv(f"./strength_covariance/model_ays/{filename}_data.csv")


def main():
    df_in = pd.read_csv("./data/models_w_props.csv")
    df_in = df_in.drop([i for i in df_in.columns if 'diamond' in i],axis=1)


    if 'disqualified' in df_in.columns:
        df_in = df_in.drop('disqualified', axis=1)

    df_in = df_in.drop([i for i in df_in.columns if 'diamond' in i], axis=1)
    df_in = df_in.drop(['thermal_expansion_coeff_bcc',
                        'surface_energy_100_bcc',
                        'surface_energy_110_bcc',
                        'surface_energy_111_bcc',
                        'surface_energy_121_bcc'], axis=1)
    
    df_clean = basic_outlier_removal(df_in)

    # create pipeline
    imput = KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)
    pca = PCA()
    #model = linear_model.LinearRegression()
    model = svm.SVR(kernel='rbf')

    pipe = Pipeline(steps=[('scale',StandardScaler()),
                            ('imp',imput),
                            ('pca',pca),
                            ('lr',model)])
    pipe = TransformedTargetRegressor(regressor = pipe,
                                            transformer = StandardScaler())

    n_bootstrap = 30

    if False:
        # set parameters 
        params_list_full = ['c44_fcc','extr_stack_fault_energy_fcc', 'unstable_stack_energy_fcc'] # best
        # params_list_full = ['c44_fcc', 'unstable_stack_energy_fcc', 'unrelaxed_formation_potential_energy_fcc'] # best w/ vac form
        # params_list_full = ['c44_sc', 'surface_energy_111_fcc', 'unstable_stack_energy_fcc']
        # params_list_full = ['c44_fcc', 'unstable_stack_energy_fcc', 'unstable_stack_energy_slip_fraction_fcc']
        # params_list_full = ['c44_fcc','c11_fcc', 'c12_fcc']
        # params_list_full = ['c11_fcc', 'unstable_stack_energy_fcc', 'unstable_twinning_energy_fcc']
        perform_bootstrap(df_clean, params_list_full, pipe, n_bootstrap, "bootstrap", title=False)

    # full model, all parameters
    if False:
        params_list = ['lattice_constant','bulk_modulus','c44','c11','c12',
                'cohesive_energy','thermal_expansion_coeff_fcc','surface_energy_100_fcc',
                'extr_stack_fault_energy','intr_stack_fault_energy','unstable_stack_energy',
                'unstable_twinning_energy','relaxed_formation_potential_energy_fcc',
                'unrelaxed_formation_potential_energy_fcc','relaxation_volume_fcc']
    
        params_list_full = filter_param_list(df_in, params_list)
        perform_bootstrap(df_clean, params_list_full, pipe, n_bootstrap, 'bootstrap_svr_all_props', title=False)



if __name__ == "__main__":
    main()