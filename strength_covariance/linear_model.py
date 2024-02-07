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
from uncertainty_quantification import r2_adj_fun
from model_selection import data_import
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from explore import import_label_dict



def pred_vs_actual_plot(df, y_pred, r2_adj, filename, title=False, factor_list = False, error_bars = False):
    y_true = df['strength_MPa']
    rmse = mean_squared_error(y_true,y_pred,squared=False)
    plt.figure(figsize = (4,3))
    p = sns.scatterplot(data=df, x=y_true,y=y_pred,hue='species')
    if error_bars == True:
        p.errorbar(y_true,y_pred, yerr = rmse, fmt='.', markersize=0.001, alpha=0.5)
    p.plot(np.linspace(min(y_true),max(y_true),50),
        np.linspace(min(y_true),max(y_true),50))
    p.set_xlabel("actual strength [MPa]")#, weight='bold',fontsize=8)  
    p.set_ylabel("predicted strength [MPa]")#, fontsize=8, weight='bold')
    if title != False:
        p.set_title(title)#,fontsize=10, weight='bold')
    note = "\nall factors"
    if factor_list != False:
        note = '\n' + factor_list 
    p.text(0.95, 0.01, f"N = {len(y_true)}\nAdjusted R\N{SUPERSCRIPT TWO} = {r2_adj:.3f}"+note,
        verticalalignment='bottom', horizontalalignment='right',
        transform=p.transAxes)#, fontsize=12, weight='bold')
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    p.set_aspect('equal', adjustable='box')
    plt.savefig(f"./strength_covariance/model_ays/{filename}.pdf", bbox_inches = 'tight')#,dpi=300)
    plt.close()

def r2_plot(r2_list, r2_adj_list, corr_list, label_dict, filename):
    corr_list = [label_dict[i] for i in corr_list]
    fig,ax = plt.subplots(figsize=(5,3))
    xloc = range(1,len(r2_list)+1)
    ax.plot(xloc,r2_list,'bx',label = f"$r^2$",)
    ax.plot(xloc,r2_adj_list,'r.',label = f"adjusted $r^2$")
    #ax.set_xlabel("Number of parameters")
    ax.set_ylabel(r"$R^2$")
    #ax.set_xticklabels(corr_list)
    ax.set_xticks(xloc, corr_list, rotation=90)
    ax.grid()
    ax.legend()
    plt.savefig(f"./strength_covariance/model_ays/{filename}.pdf",bbox_inches='tight')
    plt.close()


def y_pred_loo(pipe_in,X,y):
    y_pred = []
    for y_index in range(len(y)):
        print(f"{y_index} of {len(y)}")
        available_indexes = [j for j in range(len(y)) if j != y_index]
        X_available = X.loc[available_indexes]
        y_available = y.loc[available_indexes]
        pipe = pipe_in
        pipe.fit(X_available,y_available)
        y_pred.append(pipe.predict(X)[y_index])
    return y_pred

def y_pred_loo_w_nested_CV(pipe_in,X,y):
    # define search space
    space = dict()
    space['regressor__lr__alpha'] = [0.1,0.5,1,10,100] # strength of regularization inversely proportional to C
    #space['regressor__lr__epsilon'] = [0.01,0.05,0.1,0.5] # epsilon-tube for no penalty
    
    y_pred = []
    for y_index in range(len(y)):
        print(f"{y_index} of {len(y)}")
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

def linear_model_create(model_type):
    if model_type == "ridge":
        model = linear_model.Ridge()
    elif model_type == "lr":
        model = linear_model.LinearRegression()
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
    df, readme = data_import(clean=True)
    label_dict = import_label_dict()

    print(f"number of points initially: {len(df)}")

    params_list = ['lattice_constant',
                   'bulk_modulus','c44','c11','c12',
                   'cohesive_energy','thermal_expansion_coeff_fcc','surface_energy_100_fcc',
                   'extr_stack_fault_energy','intr_stack_fault_energy','unstable_stack_energy',
                   'unstable_twinning_energy',
                   ] # only using one surface energy, ignore vacancy params for initial plot (matches research narative)

    params_list_full = filter_param_list(df, params_list)
    #params_list_full.remove('lattice_constant_diamond')
    #params_list_full.remove('cohesive_energy_diamond')

    print(f"Number of parameters: {len(params_list_full)}")

    X_df, y = create_X_y(df, params_list_full)
    readme += f"{len(X_df.columns)} factors: {X_df.columns}\n"


    df_corr = df[['strength_MPa']+params_list_full].copy().corr(numeric_only=True).round(2)
    df_corr = abs(df_corr['strength_MPa']).sort_values(ascending=False).dropna()
    corr_list = df_corr.index.to_list()
    corr_list = corr_list[1:] # remove strength from list

    if True: #r2 plotting, before removing jammed
        # params_list_full = [i for i in params_list_full if "gb_coeff" not in i]
        pipe = linear_model_create("ridge")

        # leave out point you are predicting
        r2_list = []
        r2_all_list = []
        r2_adj_list = []
        r2_all_adj_list = []
        
        for i in range(1,len(corr_list)+1):
            print(f"factor count {i}")
            X = X_df[corr_list[:i]]
            print(f"X shape = {X.shape}, y shape = {y.shape}")
            print(f"factors = {X.columns.to_list()}")
            print("-----------")

            k = len(X.columns)
            n = len(y)
            
            pipe.fit(X,y)
            if False: #LOO R2 plot
                y_pred = y_pred_loo(pipe,X,y)
                r2_list.append(r2_score(y, y_pred))
                r2_adj_list.append(r2_adj_fun(r2_list[-1], n, k))
            
            y_pred_all = pipe.predict(X)
            r2_all_list.append(r2_score(y,y_pred_all))
            r2_all_adj_list.append(r2_adj_fun(r2_all_list[-1], n, k))

        # r2_plot(r2_list, r2_adj_list, corr_list, "linear_r2_loo_plot")  # LOO gives the strange drops/jumps due to different models each time
        r2_plot(r2_all_list, r2_all_adj_list, corr_list, label_dict, "linear_r2_plot")
        print(f"corr_list = {corr_list}")
        readme += f"\ncorr_list = {corr_list}\n"

    if True: #all factor eval, nested_cv
        # leave out point you are predicting
        pipe = linear_model_create("ridge") # had to switch to ridge due to colinearity issue/blows up for all factors
        X = X_df[corr_list]# df_clean[corr_list[:i]]
        print(f"X shape = {X.shape}, y shape = {y.shape}")
        print(f"factors = {X.columns.to_list()}")
        print("-----------")
        readme += f"\n{len(X.columns)} all factors model: {X.columns}\n"
        y_pred = y_pred_loo_w_nested_CV(pipe,X,y)
        k = len(X.columns)
        n = len(y)
        r2 = r2_score(y, y_pred)
        r2_adj = r2_adj_fun(r2, n, k)

        factor_description = f"{X.shape[1]} factors"

        # title = f"Linear model (leave one out, nested CV) using all factors"
        pred_vs_actual_plot(df, y_pred, r2_adj, "linear_all_factors_loo_nested_cv", factor_list = factor_description)

    # remove jamming
    df_clean = df[df['SF_jamming']!='yes'].reset_index()
    print(f"number of points w/o jamming: {len(df_clean)}")


    # add back in missing vacancy
    params_list.extend(['relaxed_formation_potential_energy_fcc',
                        'unrelaxed_formation_potential_energy_fcc','relaxation_volume_fcc',
                        'vacancy_migration_energy_fcc'])
    params_list_full = filter_param_list(df, params_list)

    X_df, y = create_X_y(df_clean, params_list_full)

    
    if True: #3 factor model, exclude jamming
        # still need to fix imputer to be applied at start??? no if we are assuming that data doesn't exist by practioner
        pipe = linear_model_create("lr") 
        params_short = ['c44_fcc','extr_stack_fault_energy_fcc','vacancy_migration_energy_fcc']#'unstable_stack_energy_fcc']
        factor_list = 'c44, eSFE, VME (all FCC)'
        X = X_df[params_short]
        readme += f"\n3 factor model: {X.columns}\n"
        y_pred = y_pred_loo(pipe,X,y)
        # title2 = f"Linear model (leave one out, nested CV) w/o jammed:\nc44, eSFE, uSFE (all FCC)"
        r2_adj = r2_score(y,y_pred)
        pred_vs_actual_plot(df_clean, y_pred, r2_adj, "linear_3factors_nested_cv", factor_list = factor_list, error_bars = True)

    with open(f"./strength_covariance/model_ays/linear_model_readme.txt", "w") as text_file:
        for line in readme:
            text_file.write(line)

    return

if __name__ == "__main__":
    main()