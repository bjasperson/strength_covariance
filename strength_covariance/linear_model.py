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
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from uncertainty_quantification import data_import, r2_adj_fun
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


def pred_vs_actual_plot(df, y_pred, r2_adj, title, filename):
    y_true = df['strength_MPa']
    rmse = mean_squared_error(y_true,y_pred,squared=False)
    p = sns.scatterplot(data=df, x=y_true,y=y_pred,hue='species')
    p.errorbar(y_true,y_pred, yerr = rmse, fmt='.', markersize=0.001, alpha=0.5)
    p.plot(np.linspace(min(y_true),max(y_true),50),
        np.linspace(min(y_true),max(y_true),50))
    p.set_xlabel("actual strength [MPa]")
    p.set_ylabel("predicted strength [MPa]")
    p.set_title(title,fontsize=10)
    p.text(0.95, 0.01, f"N = {len(y_true)}\nAdjusted R\N{SUPERSCRIPT TWO} = {r2_adj:.3f}",
        verticalalignment='bottom', horizontalalignment='right',
        transform=p.transAxes, fontsize=10)
    #fig = plt.figure()
    #fig.add_axes(p)
    plt.savefig(f"./strength_covariance/model_ays/{filename}.pdf",dpi=300)
    plt.close()

def r2_plot(r2_list, r2_adj_list, filename):
    fig,ax = plt.subplots()
    ax.plot(range(1,len(r2_list)+1),r2_list,'bx',label = "r2",)
    ax.plot(range(1,len(r2_list)+1),r2_adj_list,'r.',label = "r2 adj")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    plt.savefig(f"./strength_covariance/model_ays/{filename}.pdf",dpi=300)
    plt.close()


def y_pred_loo(pipe,X,y):
    y_pred = []
    for y_index in range(len(y)):
        available_indexes = [j for j in range(len(y)) if j != y_index]
        X_available = X.loc[available_indexes]
        y_available = y.loc[available_indexes]
        pipe.fit(X_available,y_available)
        y_pred.append(pipe.predict(X)[y_index])
    return y_pred

def y_pred_loo_w_nested_CV(pipe_in,X,y):
    # define search space
    space = dict()
    space['regressor__lr__alpha'] = [0.5,1,10,100] # strength of regularization inversely proportional to C
    #space['regressor__lr__epsilon'] = [0.01,0.05,0.1,0.5] # epsilon-tube for no penalty
    
    y_pred = []
    for y_index in range(len(y)):
        available_indexes = [j for j in range(len(y)) if j != y_index]
        X_train = X.loc[available_indexes]
        y_train = y.loc[available_indexes]
        cv_inner = KFold(n_splits=10, shuffle=True)
        pipe = pipe_in
        search = GridSearchCV(pipe, space, scoring='r2', n_jobs=-1, cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        print(f"best model: C = {best_model.regressor.named_steps.lr.alpha}")
        y_pred_value = best_model.predict(X).tolist()[y_index]
        y_pred.extend([y_pred_value])
    return y_pred

def linear_model_create():
    model = linear_model.Ridge()
    pca = PCA()

    pipe = Pipeline(steps=[('scale',StandardScaler()),
                           ('pca',pca),
                           ('lr',model)])
    pipe = TransformedTargetRegressor(regressor = pipe,
                                           transformer = StandardScaler())
    return pipe

def create_X_y(df, params):
    X_df = df[params]
    y = df['strength_MPa']

    # apply imputer at start; otherwise R2 doesn't increase w/ each factor added
    imput = KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)
    X_df = pd.DataFrame(imput.fit_transform(X_df), columns = imput.feature_names_in_)
    return X_df, y

def main():
    df = data_import()
    print(f"number of points initially: {len(df)}")

    params_list = ['lattice_constant','bulk_modulus','c44','c11','c12',
            'cohesive_energy','thermal_expansion_coeff_fcc','surface_energy_100_fcc',
            'extr_stack_fault_energy','intr_stack_fault_energy','unstable_stack_energy',
            'unstable_twinning_energy','relaxed_formation_potential_energy_fcc',
            'unrelaxed_formation_potential_energy_fcc','relaxation_volume_fcc',
            'gb_coeff'
            ]

    params_list_full = filter_param_list(df, params_list)
    X_df, y = create_X_y(df, params_list_full)
    pipe = linear_model_create()
    

    df_corr = df[['strength_MPa']+params_list_full].copy().corr(numeric_only=True).round(2)
    df_corr = abs(df_corr['strength_MPa']).sort_values(ascending=False).dropna()
    corr_list = df_corr.index.to_list()
    corr_list = corr_list[1:] # remove strength from list

    if True: #all factor eval, before removing jammed
        # params_list_full = [i for i in params_list_full if "gb_coeff" not in i]

        # leave out point you are predicting
        r2_list = []
        r2_all_list = []
        r2_adj_list = []
        r2_all_adj_list = []
        
        for i in range(1,len(corr_list)):
            print(f"factor count {i}")
            X = X_df[corr_list[:i]]
            print(f"X shape = {X.shape}, y shape = {y.shape}")
            print(f"factors = {X.columns.to_list()}")
            print("-----------")
            y_pred = y_pred_loo(pipe,X,y)
            pipe.fit(X,y)
            y_pred_all = pipe.predict(X)
            r2_list.append(r2_score(y, y_pred))
            r2_all_list.append(r2_score(y,y_pred_all))
            k = len(X.columns)
            n = len(y)
            r2_adj_list.append(r2_adj_fun(r2_list[-1], n, k))
            r2_all_adj_list.append(r2_adj_fun(r2_all_list[-1], n, k))

        title = f"Linear model (leave one out) using all factors"
        pred_vs_actual_plot(df, y_pred, r2_adj_list[-1], title, "linear_all_factors")
        r2_plot(r2_list, r2_adj_list,"linear_r2_loo_plot")
        r2_plot(r2_all_list, r2_all_adj_list,"linear_r2_plot")
        print(f"corr_list = {corr_list}")

    if False: #all factor eval, nested_cv
        # params_list_full = [i for i in params_list_full if "gb_coeff" not in i]

        # leave out point you are predicting
        

        X = X_df[corr_list]# df_clean[corr_list[:i]]
        print(f"X shape = {X.shape}, y shape = {y.shape}")
        print(f"factors = {X.columns.to_list()}")
        print("-----------")
        y_pred = y_pred_loo_w_nested_CV(pipe,X,y)
        k = len(X.columns)
        n = len(y)
        r2 = r2_score(y, y_pred)
        r2_adj = r2_adj_fun(r2, n, k)

        title = f"Linear model (leave one out) using all factors"
        pred_vs_actual_plot(df_clean, y_pred, r2_adj, title, "linear_all_factors_nested_cv")

    # remove jamming
    df_clean = df[df['SF_jamming']!='yes'].reset_index()
    print(f"number of points w/o jamming: {len(df_clean)}")

    X_df, y = create_X_y(df_clean, params_list_full)

    
    if True: #3 factor model, exclude jamming
        # still need to fix imputer to be applied at start??? no if we are assuming that data doesn't exist by practioner
        pipe = linear_model_create() # had to switch to ridge due to colinearity issue/blows up for all factors
        params_short = ['c44_fcc','extr_stack_fault_energy_fcc','unstable_stack_energy_fcc']
        X = X_df[params_short]
        y_pred = y_pred_loo(pipe,X,y)
        title2 = f"Linear model (leave one out) w/o jammed:\nc44, eSFE, uSFE (all FCC)"
        r2_adj = r2_score(y,y_pred)
        pred_vs_actual_plot(df_clean, y_pred, r2_adj, title2, "linear_3factors")

    if True: #3 factor model, exclude jamming, gb coeff
        # still need to fix imputer to be applied at start??? no if we are assuming that data doesn't exist by practioner
        pipe = linear_model_create() # had to switch to ridge due to colinearity issue/blows up for all factors
        params_short = ['lattice_constant_bcc', 'gb_coeff_111', 'unstable_stack_energy_fcc']
        X = X_df[params_short]
        y_pred = y_pred_loo(pipe,X,y)
        title2 = f"Linear model (leave one out) w/o jammed:\nlattice const (BCC), GB coeff (111), uSFE (FCC)"
        r2_adj = r2_score(y,y_pred)
        pred_vs_actual_plot(df_clean, y_pred, r2_adj, title2, "linear_3factors_w_gb")


    return

if __name__ == "__main__":
    main()