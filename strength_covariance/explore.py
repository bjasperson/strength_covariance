import pandas as pd
import numpy as np
from model_selection import basic_outlier_removal
import seaborn as sns
import matplotlib.pyplot as plt
import csv



def import_label_dict():
    df_labels = pd.read_csv("./strength_covariance/data_ays/label_dict.csv")
    label_dict = df_labels.to_dict(orient="records")[0]
    return label_dict


def pairplot_selected(df, factors, title, label_dict):
    factors.extend(['strength_MPa', 'species'])
    X = df[factors]
    X.columns = [label_dict[x] for x in X.columns.to_list()]
    # X.columns = [x.replace("_"," ") for x in X.columns.to_list()]
    fig = sns.pairplot(X, hue='species')
    # for ax in fig.axes.flatten():
    #     ax.set_xlabel(ax.get_xlabel(), rotation=40, ha = "right")
    fig.savefig(f"./strength_covariance/data_ays/{title}.png", dpi=300)
    return


def correlation_plot(df, title, figsize=(10, 10), annot=False):
    """plots correlation heatmap of df variables

    :param df pd.DataFrame: dataframe with labels to consider
    :param figsize: figuresize for heatmap
    "param annot: boolean, if annotated with correlation value
    """
    df_corr = df.corr().round(2)
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    colors = sns.color_palette("vlag", as_cmap=True)
    ax = sns.heatmap(df_corr, vmin=-1, vmax=1, cmap=colors, annot=annot)
    fig.add_axes(ax)
    fig.subplots_adjust(left=0.3,bottom=0.3)
    fig.savefig(f"./strength_covariance/data_ays/{title}.png", dpi=300)


def save_corr_values(df, title):
    values = df.corr(numeric_only=True)[
        'strength_MPa'].abs().sort_values(ascending=False)
    values.to_csv(f"./strength_covariance/data_ays/{title}.csv")


def run_pairplots(df_clean, label_dict):

    pairplot_selected(df_clean,
                      ['c11_fcc',
                       'c12_fcc',
                       'c44_fcc',
                       'c11-c12-c44',
                       'c11-c12+c44'],
                      'elastic_const_combos',
                      label_dict)

    pairplot_selected(df_clean,
                      ['surface_energy_100_fcc',
                       'surface_energy_110_fcc',
                       'surface_energy_111_fcc',
                       'surface_energy_121_fcc'],
                      'surface_energies',
                      label_dict)
    
    pairplot_selected(df_clean,
                      ['surface_energy_100_fcc',
                       'ideal_surface_energy_100_fcc',
                       'surface_energy_110_fcc',
                       'ideal_surface_energy_110_fcc',
                       'surface_energy_111_fcc',
                       'ideal_surface_energy_111_fcc',
                       'surface_energy_121_fcc',
                       'ideal_surface_energy_121_fcc'],
                      'surface_energies_w_ideal',
                      label_dict)

    pairplot_selected(df_clean,
                      ['extr_stack_fault_energy_fcc',
                       'intr_stack_fault_energy_fcc',
                       'unstable_stack_energy_fcc',
                        'unstable_twinning_energy_fcc'],
                      'stack_twin',
                      label_dict)

    pairplot_selected(df_clean,
                      ['lattice_constant_sc',
                       'lattice_constant_fcc',
                       'lattice_constant_bcc'],
                      'lattice_consts',
                      label_dict)

    pairplot_selected(df_clean,
                      ['cohesive_energy_bcc',
                       'cohesive_energy_sc',
                       'cohesive_energy_fcc'],
                      'cohesive_energies',
                      label_dict)

    pairplot_selected(df_clean,
                      ['c44_bcc',
                       'c44_sc',
                       'c44_fcc',
                       'thermal_expansion_coeff_fcc'],
                      'c44_thermal_exp',
                      label_dict)

    pairplot_selected(df_clean,
                      ['c11_bcc', 'c11_sc', 'c11_fcc'],
                      'c11',
                      label_dict)

    pairplot_selected(df_clean,
                      ['c12_bcc',
                       'c12_sc',
                       'c12_fcc'],
                      'c12',
                      label_dict)

    pairplot_selected(df_clean,
                      ['c44_sc',
                       'surface_energy_111_fcc',
                       'unstable_stack_energy_fcc'],
                      'highly_correlated',
                      label_dict)

    pairplot_selected(df_clean,
                      ['unstable_stack_energy_fcc',
                       'unstable_twinning_energy_fcc',
                       'unstable_twinning_energy_slip_fraction_fcc',
                       'unstable_stack_energy_slip_fraction_fcc'],
                      'unstable_stack_twin',
                      label_dict)

    pairplot_selected(df_clean,
                      ['bulk_modulus_bcc',
                       'bulk_modulus_fcc',
                       'bulk_modulus_sc'],
                      'bulk',
                      label_dict)
    
    pairplot_selected(df_clean,
                      ['c44_fcc',
                       'unstable_stack_energy_fcc',
                       'unstable_stack_energy_slip_fraction_fcc'],
                      '230805',
                      label_dict)

    pairplot_selected(df_clean,
                      ['intr_stack_fault_energy_fcc',
                       'unstable_stack_energy_fcc',
                       'unstable_twinning_energy_fcc',
                       'surface_energy_111_fcc',
                       'lattice_constant_sc',
                       'lattice_constant_fcc',
                       'cohesive_energy_fcc',
                       'c44_fcc'],
                      'big',
                      label_dict)
    
    pairplot_selected(df_clean,
                      ['relaxed_formation_potential_energy_fcc',
                       'unrelaxed_formation_potential_energy_fcc',
                       'vacancy_migration_energy_fcc',
                       'relaxation_volume_fcc'],
                       'formation energies',
                       label_dict)
    
    pairplot_selected(df_clean,
                      ['unstable_stack_energy_fcc',
                       'vacancy_migration_energy_fcc',
                       'c44_fcc',
                       'intr_stack_fault_energy_fcc'],
                       'top4_w_formation',
                       label_dict)
    
    pairplot_selected(df_clean,
                      ['bulk_modulus_fcc',
                       'c44_fcc',
                       'surface_energy_111_fcc',
                       'unstable_stack_energy_fcc',
                       'intr_stack_fault_energy_fcc'],
                       'DFT_indicator_properties',
                      label_dict)


def main():
    # import data
    label_dict = import_label_dict()
    df_in = pd.read_csv('./data/models_w_props.csv')

    save_corr_values(df_in, 'corr_initial')
    # cleanup data
    if 'disqualified' in df_in.columns:
        df_in = df_in.drop('disqualified', axis=1)

    df_in = df_in.drop([i for i in df_in.columns if 'diamond' in i], axis=1)
    df_in = df_in.drop(['thermal_expansion_coeff_bcc',
                        'surface_energy_100_bcc',
                        'surface_energy_110_bcc',
                        'surface_energy_111_bcc',
                        'surface_energy_121_bcc'], axis=1)

    # remove extreme outliers
    df_clean = basic_outlier_removal(df_in)
    save_corr_values(df_in, 'corr_clean')

    df_clean['c11-c12-c44'] = df_clean['c11_fcc'] - \
        df_clean['c12_fcc'] - df_clean['c44_fcc']
    
    df_clean['c11-c12+c44'] = df_clean['c11_fcc'] - \
        df_clean['c12_fcc'] + df_clean['c44_fcc']

    # uncomment to create pairplots
    run_pairplots(df_clean, label_dict)
    

    corr_plot_list = ['strength_MPa',
                      'intr_stack_fault_energy_fcc',
                      'unstable_stack_energy_fcc',
                      'unstable_stack_energy_slip_fraction_fcc',
                      'unstable_twinning_energy_fcc',
                      'surface_energy_111_fcc',
                      'lattice_constant_sc',
                      'lattice_constant_fcc',
                      'cohesive_energy_fcc',
                      'c44_fcc',
                      'vacancy_migration_energy_fcc',
                      'relaxed_formation_potential_energy_fcc']

    correlation_plot(df_clean[corr_plot_list],
                     "corr_plot",
                     annot=True)

    return


if __name__ == "__main__":
    main()
