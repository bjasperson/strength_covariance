import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strength_covariance.model_selection import basic_outlier_removal, filter_param_list
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from strength_covariance.model_selection import data_import
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from strength_covariance.explore import import_label_dict
import statsmodels.api as sm


def r2_adj_fun(r2,n,k):
    return (1 - ((1-r2)*(n-1)/(n-k-1)))


def pred_vs_actual_plot(df_in, 
                        y_pred, 
                        r2_adj, 
                        filename, 
                        title=False, 
                        factor_list = False, 
                        error_bars = False,
                        save_loc = "./strength_covariance/model_ays"):
    df = df_in.copy()
    df['y_pred'] = y_pred
    df = df.sort_values('species')
    y_true = df['strength_MPa']
    y_pred = df['y_pred']
    rmse = mean_squared_error(y_true,y_pred,squared=False)
    plt.figure(figsize = (4,3))
    marker_list = ['o','^','v','<','>','s','D','p','X','*','.','P']
    p = sns.scatterplot(data=df, 
                        x=y_true, 
                        y=y_pred, 
                        style = 'species', 
                        hue='species',
                        markers = marker_list[0:len(df.species.drop_duplicates())]
                        )
    if error_bars == True:
        p.errorbar(y_true,y_pred, yerr = rmse, fmt='.', markersize=0.001, alpha=0.5)

    p.plot(np.linspace(min(y_true),max(y_true),50),
           np.linspace(min(y_true),max(y_true),50)
           )
    p.set_xlabel("MD strength [MPa]")#, weight='bold',fontsize=8)  
    p.set_ylabel("predicted strength [MPa]")#, fontsize=8, weight='bold')

    if title != False:
        p.set_title(title)#,fontsize=10, weight='bold')
    
    note = f"N = {len(y_true)}\nRMSE = {rmse:.2f}\nAdjusted r\N{SUPERSCRIPT TWO} = {r2_adj:.3f}"
    height_loc = 0
    if factor_list != False:
        note += '\n' + factor_list 
        height_loc = 0.01
    p.text(0.95, height_loc, note,
        verticalalignment='bottom', horizontalalignment='right',
        transform=p.transAxes)#, fontsize=12, weight='bold')
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    p.set_aspect('equal', adjustable='box')
    plt.savefig(f"{save_loc}/{filename}.pdf", bbox_inches = 'tight')#,dpi=300)
    plt.close()

def r2_plot(r2_list, r2_adj_list, corr_list, label_dict, filename):
    corr_list = [label_dict[i] for i in corr_list]
    fig,ax = plt.subplots(figsize=(5,3))
    xloc = np.arange(1,len(r2_list)+1)
    ax.plot(xloc,r2_list,'bx',label = f"$r^2$",)
    ax.plot(xloc,r2_adj_list,'r.',label = f"adjusted $r^2$")
    ax.set_xlabel(r"$\longleftarrow$ Indicators included in model")
    ax.set_ylabel(r"$r^2$")
    #ax.set_xticklabels(corr_list)
    ax.set_xlim(0,len(xloc)+1)
    ax.set_xticks(xloc, corr_list, rotation=90)
    ax.tick_params(axis="x", labelsize = 8)
    ax.grid()
    ax.legend()

    # https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis
    ax2 = ax.twiny()
    count = np.arange(1,len(xloc)+1)
    tick_locs = np.array([1,5,10,15,20,25,30,35])
    tick_labels = ['1','5','10','15','20','25','30','35']
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels)
    #ax2.plot(xloc,,'bx',label = f"count",)
    ax2.set_xlim(0,len(xloc)+1)
    ax2.set_xlabel(r"Number of indicators included")

    plt.savefig(f"./figures/main/{filename}.pdf",bbox_inches='tight')
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

def y_pred_loo_w_nested_CV(pipe_in,
                           X,
                           y,
                           random_state = 12345):
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
        cv_inner = KFold(n_splits=10, 
                         shuffle=True,
                         random_state = random_state)
        pipe = pipe_in
        search = GridSearchCV(pipe, space, scoring='r2', n_jobs=-1, cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        print(f"best model: C = {best_model.regressor.named_steps.lr.alpha}")
        y_pred_value = best_model.predict(X).tolist()[y_index]
        y_pred.extend([y_pred_value])
    return y_pred

def linear_model_create(model_type, add_imputer = True):
    if model_type == "ridge":
        model = linear_model.Ridge()
    elif model_type == "lr":
        model = linear_model.LinearRegression()

    if add_imputer == False:
        pipe = Pipeline(steps=[('scale',StandardScaler()),
                            ('lr',model)])
    elif add_imputer == True:
        imput = KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)
        pipe = Pipeline(steps=[('scale',StandardScaler()),
                               ('imputer',imput),
                               ('lr',model)])
    pipe = TransformedTargetRegressor(regressor = pipe,
                                           transformer = StandardScaler())
    return pipe

def create_X_y(df, params, use_imputer = False):
    X_df = df[params]
    y = df['strength_MPa']

    # apply imputer at start; otherwise R2 doesn't increase w/ each factor added
    if use_imputer == True:
        imput = KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)
        X_df = pd.DataFrame(imput.fit_transform(X_df), columns = imput.feature_names_in_)
    return X_df, y

def r2_plot_code(df, 
                 params_list_full, 
                 corr_list,
                 label_dict,
                 readme
                 ):
    X_df, y = create_X_y(df, params_list_full, use_imputer = True) # use imputer at start for this plot
    readme += f"{len(X_df.columns)} factors:\n{X_df.columns.sort_values().tolist()}\n"
    pipe = linear_model_create("ridge", add_imputer = False)

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

    return readme


def all_factor_model_code(df,
                          params_list_full,
                          corr_list,
                          readme
                          ):
    # leave out point you are predicting
    pipe = linear_model_create("ridge", add_imputer = True) # had to switch to ridge due to colinearity issue/blows up for all factors
    X_df, y = create_X_y(df, params_list_full, use_imputer = False)
    X = X_df[corr_list]# df_clean[corr_list[:i]]
    print(f"X shape = {X.shape}, y shape = {y.shape}")
    print(f"factors = {X.columns.to_list()}")
    print("-----------")
    readme += f"\n{len(X.columns)} all factors model:\n{X.columns.sort_values().tolist()}\n"
    y_pred = y_pred_loo_w_nested_CV(pipe,
                                    X,
                                    y,
                                    random_state = 12345)
    k = len(X.columns)
    n = len(y)
    r2 = r2_score(y, y_pred)
    r2_adj = r2_adj_fun(r2, n, k)

    factor_description = f"{X.shape[1]} factors"

    # title = f"Linear model (leave one out, nested CV) using all factors"
    pred_vs_actual_plot(df, 
                        y_pred, 
                        r2_adj, 
                        "linear_all_factors_loo_nested_cv", 
                        # factor_list = factor_description,
                        save_loc = "./figures/main")

    return readme


def three_factor_model_code(df_clean,
                            params_short,
                            readme):


    X_df, y = create_X_y(df_clean, params_short, use_imputer = False)
    pipe = linear_model_create("lr", add_imputer = True)         
    X = X_df[params_short]
    readme += f"\n3 factor model: {X.columns.tolist()}\n"
    y_pred = y_pred_loo(pipe,X,y)
    # title2 = f"Linear model (leave one out, nested CV) w/o jammed:\nc44, eSFE, uSFE (all FCC)"
    r2_adj = r2_score(y,y_pred)
    pred_vs_actual_plot(df_clean, 
                        y_pred, 
                        r2_adj, 
                        "linear_3factors_nested_cv", 
                        # factor_list = factor_list, 
                        error_bars = False,
                        save_loc = "./figures/main")
    
    readme += "\n\n---------------------\n3 factor leave-one-out results\n\n"
    readme += pd.DataFrame({"strength":y,
                            "predicted":y_pred,
                            "error":(y - y_pred),
                            "rel error":(y - y_pred)/y_pred,
                            "species":df_clean.species}
                            ).sort_values(['strength','species']).to_string()

    return readme

def main():
    df, readme = data_import(clean=True,
                             random_state = 12345)
    label_dict = import_label_dict()

    print(f"number of points initially: {len(df)}")

    # ignore vacancy params for initial plot (matches research narative)
    readme += "\n\n--------linear model----------\n"\
              "all factor linear model does not include "\
              "vacancy formation/migration energies at this point in manuscript\n"
    params_list = ['lattice_constant',
                   'bulk_modulus','c44','c11','c12',
                   'cohesive_energy','thermal_expansion_coeff_fcc','surface_energy',
                   'extr_stack_fault_energy','intr_stack_fault_energy','unstable_stack_energy',
                   'unstable_twinning_energy',
                   ] 

    params_list_full = filter_param_list(df, params_list)

    print(f"Number of parameters: {len(params_list_full)}")

    df_corr = df[['strength_MPa']+params_list_full].copy().corr(numeric_only=True).round(2)
    df_corr = abs(df_corr['strength_MPa']).sort_values(ascending=False).dropna()
    corr_list = df_corr.index.to_list()
    corr_list = corr_list[1:] # remove strength from list

    readme = r2_plot_code(df,
                          params_list_full,
                          corr_list,
                          label_dict,
                          readme) #r2 plotting, before removing jammed
        
    readme = all_factor_model_code(df,
                                   params_list_full,
                                   corr_list,
                                   readme)

    # remove jamming
    df_clean = df[df['SF_jamming']!='yes'].reset_index()
    print(f"number of points w/o jamming: {len(df_clean)}")


    # add back in missing vacancy
    params_list.extend(['relaxed_formation_potential_energy_fcc',
                        'unrelaxed_formation_potential_energy_fcc','relaxation_volume_fcc',
                        'vacancy_migration_energy_fcc'])
    params_list_full = filter_param_list(df, params_list)

    params_short = ['vacancy_migration_energy_fcc',
                    'surface_energy_111_fcc',
                    'lattice_constant_fcc']

    readme = three_factor_model_code(df_clean,
                                     params_short,
                                     readme)
 
    with open(f"./strength_covariance/model_ays/linear_model_readme.txt", "w") as text_file:
        for line in readme:
            text_file.write(line)


if __name__ == "__main__":
    main()