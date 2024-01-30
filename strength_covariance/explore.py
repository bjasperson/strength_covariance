import pandas as pd
import numpy as np
from model_selection import basic_outlier_removal, filter_param_list
import seaborn as sns
import matplotlib.pyplot as plt
import csv



def import_label_dict():
    df_labels = pd.read_csv("./strength_covariance/data_ays/label_dict.csv")
    label_dict = df_labels.to_dict(orient="records")[0]
    return label_dict


def pairplot_selected(df, factors, title, label_dict, corner = False, height=1.5):
    factors.extend(['strength_MPa', 'species'])
    X = df[factors]
    X.columns = [label_dict[x] for x in X.columns.to_list()]
    sns.set(style="whitegrid")#,font_scale=1.25)
    # X.columns = [x.replace("_"," ") for x in X.columns.to_list()]
    #fig,ax = plt.subplots(figsize = (3,3))
    g = sns.pairplot(X, hue='species', corner = corner, height=height)#,
                       #plot_kws={"s":100})
    # for ax in fig.axes.flatten():
    #     ax.set_xlabel(ax.get_xlabel(), rotation=40, ha = "right")
    sns.move_legend(g,"upper right",bbox_to_anchor=(0.85,1))
    g.savefig(f"./strength_covariance/data_ays/{title}.pdf")#, dpi=300)
    plt.close()
    return


def correlation_df(df, label_dict):
    df_corr = df.corr().round(2)
    columns = df_corr.columns.to_list()
    columns = [label_dict[x] for x in columns]
    df_corr.columns = columns
    
    df_index = df_corr.index.to_list()
    df_index = [label_dict[x] for x in df_index]
    df_corr.index = df_index

    order = df_corr['Strength'].sort_values(ascending=False).index.to_list()
    df_corr = df_corr[order].reindex(order)
    return df_corr


def correlation_plot(df, title, label_dict, figsize=(10, 10), annot=False, lower = False):
    """plots correlation heatmap of df variables

    :param df pd.DataFrame: dataframe with labels to consider
    :param figsize: figuresize for heatmap
    "param annot: boolean, if annotated with correlation value
    """

    df_corr = correlation_df(df, label_dict)
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    colors = sns.color_palette("vlag", as_cmap=True)
    if lower == True:
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        mask[np.diag_indices_from(mask)] = False
    else:
        mask = np.zeros_like(df_corr)
    ax = sns.heatmap(df_corr, mask = mask, vmin=-1, vmax=1, cmap=colors, annot=annot,
                     cbar_kws = dict(use_gridspec=False,location='top'))
    ax.set_facecolor('white')
    fig.add_axes(ax)
    #fig.subplots_adjust(left=0.3,bottom=0.3)
    fig.savefig(f"./strength_covariance/data_ays/{title}.pdf", bbox_inches = 'tight')#, dpi=300)


def save_corr_values(df, title):
    values = df.corr(numeric_only=True)[
        'strength_MPa'].abs().sort_values(ascending=False)
    values.to_csv(f"./strength_covariance/data_ays/{title}.csv")


def run_pairplots(df_clean, label_dict):

    pairplot_selected(df_clean,
                      ['c11_fcc',
                       'c12_fcc',
                       'c44_fcc',
                       #'c11-c12-c44',
                       'c11-c12+c44'],
                      'elastic_const_combos',
                      label_dict)
    
    pairplot_selected(df_clean,
                      ['c11_fcc',
                       'c12_fcc',
                       'c44_fcc',
                       'lattice_constant_sc',
                       'lattice_constant_fcc',
                       'lattice_constant_bcc'],
                       'elastic_lattice_const',
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
                       'intr_stack_fault_energy_fcc',
                       #'gb_coeff_111',
                       ],
                       'pairplot_top_factors',
                       label_dict)
    
    pairplot_selected(df_clean,
                      ['bulk_modulus_fcc',
                       'c44_fcc',
                       'surface_energy_111_fcc',
                       'unstable_stack_energy_fcc',
                       'intr_stack_fault_energy_fcc'],
                       'DFT_indicator_properties',
                      label_dict)
    
    # pairplot_selected(df_clean,
    #                   ['gb_coeff_001',
    #                    'gb_coeff_110',
    #                    'gb_coeff_111',
    #                    'gb_coeff_112'
    #                    ],
    #                    'gb_coeff',
    #                   label_dict)
    

def manuscript_plots(df_clean, label_dict):
    param_list = ['vacancy_migration_energy_fcc',
                  'c44_fcc',
                  'relaxed_formation_potential_energy_fcc',
                  'unstable_stack_energy_fcc',
                  'intr_stack_fault_energy_fcc'
                    ]
    
    #sns.set(font_scale=1.5)
    
    pairplot_selected(df_clean,
                    param_list,
                    'manuscript_pairplot',
                    label_dict,
                    height=1.5,
                    corner = True)
    
    #sns.set(font_scale=1.5)
    correlation_plot(df_clean[param_list],
                    "corr_plot_manuscript",
                    label_dict,
                    figsize=(3,4),
                    annot = True,
                    lower = True)
    
    #sns.set(font_scale = 1)

    # initial attempt to combine pairplot and corr matrix.
    # not working yet
    # https://stackoverflow.com/questions/63416894/correlation-values-in-pairplot
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    # checkout ax.set_facecolor
    # df_corr = correlation_df(df_clean[param_list],label_dict)
    # g = sns.PairGrid(df_clean[param_list])
    # g.map_diag(sns.distplot)
    # g.map_lower(sns.regplot)
    # g.map_upper(sns.heatmap(df_corr))
    # plt.show()
    return


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
    save_corr_values(df_clean, 'corr_clean')

    # df_clean['c11-c12-c44'] = df_clean['c11_fcc'] - \
    #     df_clean['c12_fcc'] - df_clean['c44_fcc']
    
    df_clean['c11-c12+c44'] = df_clean['c11_fcc'] - \
        df_clean['c12_fcc'] + df_clean['c44_fcc']

    # uncomment to run all pairplots
    # run_pairplots(df_clean, label_dict)

    # pairplot_selected(df_clean,
    #                 ['gb_coeff_001',
    #                 'gb_coeff_110',
    #                 'gb_coeff_111',
    #                 'gb_coeff_112'],
    #                 'gb_coeff',
    #                 label_dict)

    corr_plot_list = ['strength_MPa',
                      'intr_stack_fault_energy_fcc',
                      'unstable_stack_energy_fcc',
                      'unstable_stack_energy_slip_fraction_fcc',
                      'unstable_twinning_energy_fcc',
                      'surface_energy_110_fcc',
                      'lattice_constant_sc',
                      'lattice_constant_fcc',
                      'cohesive_energy_fcc',
                      'c44_fcc',
                      'vacancy_migration_energy_fcc',
                      'relaxed_formation_potential_energy_fcc']

    correlation_plot(df_clean[corr_plot_list],
                     "corr_plot",
                     label_dict,
                     annot=True)
    
    params_list = ['lattice_constant', 
                'bulk_modulus', 'c11', 'c12', 'c44',
                'cohesive_energy_fcc', 'thermal_expansion_coeff_fcc',
                'surface_energy_100_fcc',
                'surface_energy_110_fcc',
                'surface_energy_111_fcc',
                'surface_energy_121_fcc',
                'extr_stack_fault_energy', 
                'intr_stack_fault_energy',
                'unstable_stack_energy', 'unstable_twinning_energy',
                'relaxed_formation_potential_energy_fcc', #includes unrelaxed
                'vacancy_migration_energy_fcc',
                'relaxation_volume_fcc',
                # 'gb_coeff_001','gb_coeff_110','gb_coeff_111','gb_coeff_112',
                ]

    params_list_full = filter_param_list(
        df_clean, params_list, ['strength_MPa'])
    
    params_list_full = [i for n, i in enumerate(params_list_full) if i not in params_list_full[:n]]

    
    correlation_plot(df_clean[params_list_full],
                    "corr_plot_full",
                    label_dict,
                    figsize=(4,4),
                    annot=False)
    
    manuscript_plots(df_clean, label_dict)

    return


if __name__ == "__main__":
    main()
