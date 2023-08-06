import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn import linear_model, svm
from ast import literal_eval
import seaborn as sns


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

    return params_list_full


def factor_select_cv(X, y, pipe, n_factor_max=2, cv=5, scoring='r2'):
    # return list of parameters w/ cv score
    factor_list = X.columns.to_list()
    subsets = []
    for n in range(1, (n_factor_max+1)):
        for subset in combinations(factor_list, n):
            subsets.append(list(subset))

    cv_score_mean = []
    cv_score_std = []
    for subset in subsets:
        print('current subset: ', subset)
        score = cross_val_score(pipe, X[subset], y, cv=cv, scoring=scoring)
        print('score mean: ', np.mean(score))
        cv_score_mean.append(np.mean(score))
        cv_score_std.append(np.std(score))

    df_results = pd.DataFrame({'factors': subsets,
                               'cv_score': cv_score_mean,
                               'cv_score_std': cv_score_std})
    df_results = df_results.sort_values('cv_score', ascending=False)

    return df_results


def factor_percent_usage(df_results, N_lines, title):
    factor_list_in = df_results['factors'].iloc[:N_lines]
    #factor_list = [literal_eval(i) for i in factor_list_in] #only needed when importing a csv
    factor_list = [i for i in factor_list_in]
    factor_list_combined = [j for i in factor_list for j in i]
    df_factors = pd.DataFrame(factor_list_combined)
    fig = plt.figure()
    ax = df_factors.value_counts().plot.bar()
    fig.add_axes(ax)
    fig.subplots_adjust(bottom=0.6)
    fig.savefig(f"./strength_covariance/model_ays/{title}.png", dpi=300)
    return


def main():
    df_in = pd.read_csv("./data/models_w_props.csv")

    df_in = df_in.drop([i for i in df_in.columns if 'diamond' in i], axis=1)
    df_in = df_in.sample(frac=1)  # shuffle

    params_list = ['lattice_constant',
                   'bulk_modulus', 'c11', 'c12', 'c44',
                   'cohesive_energy_fcc', 'thermal_expansion_coeff_fcc',
                   'surface_energy_100_fcc',
                   'extr_stack_fault_energy',
                   'unstable_stack_energy']

    params_list_full = filter_param_list(df_in, params_list)

    df_clean = basic_outlier_removal(df_in)

    X = df_clean[params_list_full]
    y = df_clean['strength_MPa']

    imput = KNNImputer(n_neighbors=2, weights="uniform",
                       keep_empty_features=True)
    pca = PCA()
    #model = linear_model.LinearRegression()
    model = svm.SVR(kernel='rbf')

    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('imp', imput),
                           ('pca', pca),
                           ('lr', model)])
    pipe = TransformedTargetRegressor(regressor=pipe,
                                      transformer=StandardScaler())

    n_factor_max = 3

    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    df_results = factor_select_cv(
        X, y, pipe, n_factor_max=n_factor_max, cv=cv, scoring='neg_root_mean_squared_error')
    df_results.to_csv("./strength_covariance/model_ays/kfold_models.csv")
    # df_results = pd.read_csv("./strength_covariance/model_ays/kfold_models.csv")
    factor_percent_usage(df_results, 100, 'kfold_factor_usage')

    # loocv = LeaveOneOut()
    # df_results_loocv = factor_select_cv(
    #     X, y, pipe, n_factor_max=n_factor_max, cv=loocv, scoring='neg_root_mean_squared_error')
    # df_results_loocv.to_csv("./strength_covariance/model_ays/loocv_models.csv")
    # # df_results_loocv = pd.read_csv("./strength_covariance/model_ays/loocv_models.csv")
    # factor_percent_usage(df_results_loocv, 100, 'loocv_factor_usage')
    return


if __name__ == "__main__":
    main()
