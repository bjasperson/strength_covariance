import pandas as pd
import numpy as np
from model_selection import basic_outlier_removal, filter_param_list, data_import, get_factor_list
import seaborn as sns
import matplotlib.pyplot as plt
import csv



def import_label_dict():
    df_labels = pd.read_csv("./strength_covariance/data_ays/label_dict.csv")
    label_dict = df_labels.to_dict(orient="records")[0]
    return label_dict


def pairplot_selected(df, 
                      factors, 
                      title, 
                      label_dict, 
                      corner = False, 
                      height=1.5,
                      save_location = "./strength_covariance/data_ays",
                      marker_size = 75):
    if 'strength_MPa' not in factors:
        factors.extend(['strength_MPa'])
    if 'species' not in factors:
        factors.extend(['species'])
    X = df[factors]
    X.columns = [label_dict[x] for x in X.columns.to_list()]
    X = X.sort_values('species')
    sns.set(style="whitegrid")#,font_scale=1.25)
    # X.columns = [x.replace("_"," ") for x in X.columns.to_list()]
    #fig,ax = plt.subplots(figsize = (3,3))
    marker_list = ['o','^','v','<','>','s','D','p','X','*','.','P']
    g = sns.pairplot(X, hue='species', markers = marker_list[0:len(df.species.drop_duplicates())], corner = corner, height=height,
                       plot_kws={"s":marker_size})
    # for ax in fig.axes.flatten():
    #     ax.set_xlabel(ax.get_xlabel(), rotation=40, ha = "right")
    sns.move_legend(g,"upper right",bbox_to_anchor=(0.85,1))
    g.savefig(f"{save_location}/{title}.pdf")#, dpi=300)
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


def correlation_plot(df, 
                     title, 
                     label_dict, 
                     labelsize = 8,
                     figsize=(10, 10), 
                     annot_fontsize = 12,
                     annot=False, lower = False,
                     custom_order = False,
                     save_location = './strength_covariance/data_ays'):
    """plots correlation heatmap of df variables

    :param df pd.DataFrame: dataframe with labels to consider
    :param figsize: figuresize for heatmap
    "param annot: boolean, if annotated with correlation value
    """

    df_corr = correlation_df(df, label_dict)
    if custom_order != False:
        df_corr = df_corr[custom_order]
        df_corr = df_corr.loc[custom_order]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    colors = sns.color_palette("vlag", as_cmap=True)
    if lower == True:
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        mask[np.diag_indices_from(mask)] = False
    else:
        mask = np.zeros_like(df_corr)
    ax = sns.heatmap(df_corr, 
                     mask = mask, 
                     xticklabels = True,
                     yticklabels = True,
                     fmt = '.1f',
                     vmin=-1, 
                     vmax=1, 
                     cmap=colors, 
                     annot=annot,
                     annot_kws = {"fontsize":annot_fontsize},
                     cbar_kws = dict(use_gridspec=False,
                                     location='top',
                                     shrink=0.5))
    ax.set_facecolor('white')
    ax.grid(False)
    ax.tick_params(labelsize=labelsize)
    fig.add_axes(ax)
    #fig.subplots_adjust(left=0.3,bottom=0.3)
    fig.savefig(f"{save_location}/{title}.pdf", bbox_inches = 'tight')#, dpi=300)


def save_corr_values(df, title):
    values = df.corr(numeric_only=True).round(decimals = 4)
    values.to_csv(f"./strength_covariance/data_ays/{title}_all.csv")
    values['strength_MPa'].abs().sort_values(ascending=False).to_csv(f"./strength_covariance/data_ays/{title}.csv")


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
    param_list = ['c44_fcc',
                  'relaxed_formation_potential_energy_fcc',
                  'unstable_stack_energy_fcc',
                  'intr_stack_fault_energy_fcc'
                    ]
    
    save_location = "figures/main"
    
    #sns.set(font_scale=1.5)
    
    pairplot_selected(df_clean,
                    param_list,
                    'manuscript_pairplot',
                    label_dict,
                    height=1.5,
                    corner = True,
                    save_location = save_location)
    
    #sns.set(font_scale=1.5)
    correlation_plot(df_clean[param_list],
                    "corr_plot_manuscript",
                    label_dict,
                    figsize=(3,4),
                    annot_fontsize = 8,
                    annot = True,
                    lower = True,
                    custom_order=['C44-FCC','rVFPE-FCC','uSFE-FCC','iSFE-FCC','Strength'],
                    save_location = save_location)
    
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


def supplemental_plots(df_clean, label_dict):
    save_location = "figures/si"
    params_list_full = get_factor_list(df_clean)
    params_list_full.append('strength_MPa')

    correlation_plot(df_clean[params_list_full],
                    "corr_plot_full",
                    label_dict,
                    labelsize = 7,
                    figsize=(6.5,9),
                    annot=True,
                    annot_fontsize = 5,
                    save_location = save_location)
       
    param_list = ['vacancy_migration_energy_fcc',
                  'c44_fcc',
                  'surface_energy_100_fcc',
                  'relaxed_formation_potential_energy_fcc',
                  'unstable_stack_energy_fcc',
                  'lattice_constant_bcc'
                  ]

    pairplot_selected(df_clean,
                    param_list,
                    'pp_with_jammed',
                    label_dict,
                    height=1.5,
                    corner = True,
                    save_location = save_location,
                    marker_size = 50)

    df_clean = df_clean[df_clean['SF_jamming']!='yes'].reset_index(drop = True)

    pairplot_selected(df_clean,
                    param_list,
                    'pp_wo_jammed',
                    label_dict,
                    height=1.5,
                    corner = True,
                    save_location = save_location,
                    marker_size = 50)
    
    pairplot_selected(df_clean,
                      ['surface_energy_100_fcc',
                       'surface_energy_110_fcc',
                       'surface_energy_111_fcc',
                       'surface_energy_121_fcc'],
                      'pp_surf_energies',
                      label_dict,
                      height=1.5,
                      corner = True,
                      save_location = save_location)
    
    pairplot_selected(df_clean,
                      ['unstable_twinning_energy_fcc',
                       'unstable_stack_energy_fcc',
                       'intr_stack_fault_energy_fcc',
                       'extr_stack_fault_energy_fcc'
                       ],
                      'pp_stack_twin',
                      label_dict,
                      height=1.5,
                      corner = True,
                      save_location = save_location)


def main():
    # import data
    label_dict = import_label_dict()
    df_in, readme = data_import(clean = False)

    save_corr_values(df_in, 'corr_initial')

    # remove extreme outliers
    df_clean = basic_outlier_removal(df_in)
    save_corr_values(df_clean, 'corr_clean')
    
    df_clean['c11-c12+c44'] = df_clean['c11_fcc'] - \
        df_clean['c12_fcc'] + df_clean['c44_fcc']

    # uncomment to run all pairplots
    # run_pairplots(df_clean, label_dict)
    
    manuscript_plots(df_clean, label_dict)
    supplemental_plots(df_clean, label_dict)

    return


if __name__ == "__main__":
    main()
