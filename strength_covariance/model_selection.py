import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate, LeaveOneOut, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import linear_model, svm
from ast import literal_eval
import seaborn as sns


def data_import(clean=False,
                data_path = "./data/models_w_props.csv",
                readme = "",
                random_state = 12345):
    
    df = pd.read_csv(data_path)
    readme += f"data import:\nInitial shape: {df.shape}\n"

    # count initial factor list
    factor_list = get_factor_list(df)
    readme += f"initial factor list = {len(factor_list)}\n"

    # remove very sparse columns
    drop_list = ['bulk_modulus_diamond','c44_diamond','c12_diamond','c11_diamond']
    df = df.drop(drop_list,axis=1)
    readme += f"removed {len(drop_list)} very sparse columns: {drop_list}\n"
    
    if 'disqualified' in df.columns:
        df = df.drop('disqualified', axis=1)

    # drop empty columns
    drop_cols = []
    for col in df.columns.tolist():
        if df[col].isnull().all():
            drop_cols.append(col)
    df = df.drop(drop_cols,axis=1)
    readme += f"removed {len(drop_cols)} empty columns: {drop_cols}\n"

    if clean==True:
        df = basic_outlier_removal(df)
        readme += f"cleaned data\n"

    factor_list = get_factor_list(df)
    readme += f"final factor count is {len(factor_list)}: {factor_list}\n"


    df = df.sample(frac=1, random_state = random_state).reset_index(drop=True)  # shuffle
    readme += f"shuffled\n"
    return df, readme


def get_factor_list(df):
    factor_list = df.columns.tolist()
    factor_list = [i for i in factor_list if ("_fcc" in i) or ("_bcc" in i) or ("_sc" in i) or ("_diamond" in i) or ("_hcp" in i)]
    return factor_list


def basic_outlier_removal(df):
    """remove extreme canonical property predictions

    :param df: dataframe with outliers
    :return: dataframe without outliers
    """

    columns = df.columns.to_list()
    # get list of bulk_modulus
    bulk_list = [i for i in columns if "bulk" in i]
    # get list of elastic constants
    elastic_const_list = [i for i in columns if "c11" in i]
    elastic_const_list.extend([i for i in columns if "c12" in i])
    elastic_const_list.extend([i for i in columns if "c44" in i])
    # get list of cohesive_energy
    cohesive_energy_list = [i for i in columns if "cohesive_energy" in i]
    # get list of stack_fault_energy
    stack_energy_list = [i for i in columns if "stack_fault_energy" in i]

    # set limits and remove extreme outliers
    df[bulk_list] = df[bulk_list][df[bulk_list] < 100000]
    df[elastic_const_list] = df[elastic_const_list][df[elastic_const_list] < 60000]
    df[cohesive_energy_list] = df[cohesive_energy_list][df[cohesive_energy_list] < 45]
    df[stack_energy_list] = df[stack_energy_list][df[stack_energy_list] < 0.45]
    return df


def filter_param_list(df, base_labels, specific_items=""):
    """generate filtered list of parameters

    :param df pd.DataFrame: dataframe with columns to consider
    :param base_labels list: base label strings to consider
    :param specific_items list: specific strings to include
    :return: list of specific label strings of parameters 
    """
    params_list_full = []
    params_list_full.extend(specific_items)

    for i in base_labels:
        current_list = [j for j in df.columns if i in j]
        params_list_full.extend(current_list)

    params_list_full = list(set(params_list_full)) # remove duplicates

    return params_list_full


def factor_select_cv(X, y, pipe, n_factor_max=2, cv=5):
    # return list of parameters w/ cv score
    factor_list = X.columns.to_list()
    subsets = []
    for n in range(1, (n_factor_max+1)):
        for subset in combinations(factor_list, n):
            subsets.append(list(subset))

    cv_score_rmse_mean = []
    cv_score_r2_mean = []
    for i, subset in enumerate(subsets):
        if i%100 == 0:
            print(f'{i} of {len(subsets)}')
        scoring = {'neg_rmse':'neg_root_mean_squared_error',
                   'r2':'r2'}
        score = cross_validate(pipe, X[subset], y, cv=cv, scoring=scoring, n_jobs = -1)
        cv_score_rmse_mean.append(np.mean(score['test_neg_rmse']))
        cv_score_r2_mean.append(np.mean(score['test_r2']))

    df_results = pd.DataFrame({'factors': subsets,
                               'r2_cv_score': cv_score_r2_mean,
                               'rmse_cv_score': cv_score_rmse_mean})
    df_results = df_results.sort_values('r2_cv_score', ascending=False)

    return df_results


def factor_percent_usage(df_results, N_lines, title):
    factor_list_in = df_results['factors'].iloc[:N_lines]
    factor_list = [i for i in factor_list_in]
    if isinstance(factor_list[0], str):
        factor_list = [literal_eval(i) for i in factor_list_in] #only needed when importing a csv
    factor_list_combined = [j for i in factor_list for j in i]
    df_factors = pd.DataFrame(factor_list_combined)
    fig = plt.figure()
    ax = df_factors.value_counts().plot.bar()
    fig.add_axes(ax)
    fig.subplots_adjust(bottom=0.6)
    fig.savefig(f"./strength_covariance/model_ays/{title}.png", dpi=300)
    return df_factors.value_counts().rename_axis('factor').reset_index(name=title)

def create_factor_select_plot(df_merge, filename, label_dict, model_dict):
    model_list = [i for i in model_dict.keys()]
    
    df_corr = df_merge.corr(numeric_only = True).round(2)
    abs_strength_corr = abs(df_corr['strength_MPa']).sort_values(ascending=False)
    abs_strength_corr.index.name = "factor"
    abs_strength_corr = abs_strength_corr.rename('corr_coeff')
    abs_strength_corr = abs_strength_corr.drop('strength_MPa')

    to_include = []
    for model in model_list:
        df_results = pd.read_csv(f"./strength_covariance/model_ays/{model}.csv", index_col=0)
        df_factor_count = factor_percent_usage(df_results, 100, model_dict[model])
        df_factor_count = df_factor_count.set_index('factor')
        max_count = df_factor_count.max()[model_dict[model]]
        min_count = df_factor_count.min()[model_dict[model]]
        df_factor_count = (df_factor_count - min_count)/(max_count - min_count)
        abs_strength_corr = pd.merge(df_factor_count, abs_strength_corr, left_index = True, right_index = True, how='outer').fillna(0)
        to_include.extend(abs_strength_corr[abs_strength_corr[model_dict[model]]>0].index.tolist())

    to_include = list(set(to_include))
    df = abs_strength_corr.loc[to_include]
    
    df = df.rename(columns = {"corr_coeff":"Correlation\nCoefficient"})
    df = df.rename(columns = model_dict)
    
    df = df.sort_values("Correlation\nCoefficient", ascending=False)
    factor_select_plotting(df.iloc[:15,:], label_dict, filename+"_corr", width = 0.25)

    df = df.sort_values(model_dict[model_list[-1]], ascending=False)
    factor_select_plotting(df.iloc[:15,:], label_dict, filename+"_count", width = 0.25)

def factor_select_plotting(df, label_dict, filename, width = 0.125, size = (7,3)): 
    cols = df.columns

    x = np.arange(len(df))    
    multiplier = 0


    #plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
    #plt.rc('ytick', labelsize=16) #fontsize of the y tick labels

    fig, ax = plt.subplots(figsize=size)
    for attribute in cols:
        offset = width * multiplier
        scaled_value = df[attribute]
        #scaled_value = (df[attribute]-min(df[attribute]))/(max(df[attribute])-min(df[attribute]))
        #scaled_value[scaled_value==0.0] = 0.01 # set small value for plotting
        ax.bar(x+offset,scaled_value,width, label=attribute)
        #ax.set_ylabel("$R^2$ improvement (percentage)")

        #axs[1].set_ylabel('Standardized Regression Coefficients (absolute)')
        multiplier += 1
    #ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    x_labels =  [label_dict[x] for x in df.index.to_list()]
    #ax.set_ylabel("Correlation Coefficient or Normalized Factor Usage\n(Value - Min Value)/(Max Value - Min Value)",fontsize=16)
    ax.set_ylabel("Correlation Coefficient \nor Normalized Factor Usage")#,fontsize=16)
    tick_loc = (len(cols)*width/2-width/2)
    ax.set_xticks(x + tick_loc, x_labels, rotation = 90)
    ax.legend(bbox_to_anchor = (0,1,1,1), loc="lower center", mode="expand", ncol = 4,fontsize=8)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(0.5+x+tick_loc))
    ax.xaxis.grid(visible=True, which="minor")

    fig.savefig(f"./strength_covariance/model_ays/{filename}.pdf", bbox_inches = 'tight')#dpi=300)

    return

def top5_table(label_dict, model_dict, sort_by):
    model_list = [i for i in model_dict.keys()]

    for model in model_list:
        df_results = pd.read_csv(f"./strength_covariance/model_ays/{model}.csv", index_col=0)
        df_results = df_results.sort_values(sort_by, ascending=False)
        factor_list = [literal_eval(i) for i in df_results['factors'].iloc[0:5]]
        df5 = pd.DataFrame(factor_list)
        df5 = df5.replace(label_dict)
        df5.to_csv(f"strength_covariance/model_ays/{model}_top5.csv")
    return

def model_create(model_type):
    imput = KNNImputer(n_neighbors=2, weights="uniform",
                       keep_empty_features=True)

    if model_type == "lr":
        model = linear_model.LinearRegression()
    elif model_type == "svr":
        model = svm.SVR(kernel='rbf')
    elif model_type == "ridge":
        model = linear_model.Ridge()

    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('imp', imput),
                           ('lr', model)])
    pipe = TransformedTargetRegressor(regressor=pipe,
                                      transformer=StandardScaler())

    return pipe

def main():
    from explore import import_label_dict

    df_clean, readme = data_import(clean=True)
    label_dict = import_label_dict()

    params_list = ['lattice_constant',
                   'bulk_modulus', 'c11', 'c12', 'c44',
                   'cohesive_energy', # highly correlated with other CohEng; LM uses all
                   'thermal_expansion_coeff_fcc',
                   'surface_energy',
                   'extr_stack_fault_energy',
                   'intr_stack_fault_energy', # highly correlated with extr SFE
                   'unstable_stack_energy', 
                   'unstable_twinning_energy', #highly correlated with unstable SFE
                   'relaxed_formation_potential_energy_fcc', #includes unrelaxed
                   'unrelaxed_formation_potential_energy_fcc', 
                   'vacancy_migration_energy_fcc',
                   'relaxation_volume_fcc'
                   ]

    params_list_full = filter_param_list(df_clean, params_list)
    print(f"number of factors: {len(params_list_full)}\n")
    print(f"factors list: {params_list_full}")

    # remove jammed; significantly changes importance of uSFE in model selection for SVR
    print(f"number of points before removing jamming: {len(df_clean)}")
    df_clean = df_clean[df_clean['SF_jamming']!='yes'].reset_index()
    print(f"number of points w/o jamming: {len(df_clean)}")

    X = df_clean[params_list_full]
    y = df_clean['strength_MPa']

    n_factor_max = 3
    if False: # this is the one used for the manuscript
        pipe = model_create('lr')
        cv = RepeatedKFold(n_splits=10, n_repeats=5)
        df_results = factor_select_cv(
            X, y, pipe, n_factor_max=n_factor_max, cv=cv)
        df_results.to_csv("./strength_covariance/model_ays/kfold_lr_models.csv")
        factor_percent_usage(df_results, 100, 'kfold_lr_factor_usage')

    if False:
        pipe = model_create('svr')
        cv = RepeatedKFold(n_splits=10, n_repeats=5)
        df_results = factor_select_cv(
            X, y, pipe, n_factor_max=n_factor_max, cv=cv)
        df_results.to_csv("./strength_covariance/model_ays/kfold_svr_models.csv")
        factor_percent_usage(df_results, 100, 'kfold_svr_factor_usage')

    if False:
        pipe = model_create('ridge')
        cv = RepeatedKFold(n_splits=10, n_repeats=5)
        df_results = factor_select_cv(
            X, y, pipe, n_factor_max=n_factor_max, cv=cv)
        df_results.to_csv("./strength_covariance/model_ays/kfold_ridge_models.csv")
        factor_percent_usage(df_results, 100, 'kfold_ridge_factor_usage')

    if False: #LOOCV
        pipe = model_create('ridge')
        loocv = LeaveOneOut()
        df_results_loocv = factor_select_cv(
            X, y, pipe, n_factor_max=n_factor_max, cv=loocv)
        df_results_loocv.to_csv("./strength_covariance/model_ays/loocv_ridge_models.csv")
        factor_percent_usage(df_results_loocv, 100, 'loocv_ridge_factor_usage')

    if False: # load results and plot without rerunning kFold or LOO
        df_results = pd.read_csv("./strength_covariance/model_ays/kfold_models.csv")
        factor_percent_usage(df_results, 100, 'kfold_factor_usage')
        df_results_loocv = pd.read_csv("./strength_covariance/model_ays/loocv_models.csv")
        factor_percent_usage(df_results_loocv, 100, 'loocv_factor_usage')

    if False: # obsolete plotting for non-linear models
        model_dict = {"kfold_lr_models":"Freq. of occurrence\ntop LR models",
                      "kfold_svr_models":"Freq. of occurrence\ntop SVR models"}
        create_factor_select_plot(df_clean, "factor_importance", label_dict, model_dict)

        model_dict = {"kfold_ridge_models":"Freq. of occurrence\ntop ridge models",
                      "kfold_svr_models":"Freq. of occurrence\ntop SVR models"}
        create_factor_select_plot(df_clean, "factor_importance_2", label_dict, model_dict)

    if True:
        model_dict = {"kfold_lr_models":"Freq. of occurrence\ntop LR models"}
        create_factor_select_plot(df_clean, "factor_importance_3", label_dict, model_dict)
        top5_table(label_dict, model_dict, "rmse_cv_score")
    return


if __name__ == "__main__":
    main()
