import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

# Global styling
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = 'gray'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.6
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['figure.titlesize'] = 40
mpl.rcParams['axes.titlesize'] = 35
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

models = ['lgssm_N_32_K_15_D_3_P_500_fix_v',  'sir_N_32_K_15_D_2_P_500'] # 'svm_N_32_K_15_D_3_P_500', 'earthquake_N_32_K_15_D_3_P_500',

def boxplot_rmse(model):
    # Load the RMSE data
    rmse_path = f'outputs/{model}/rmse.csv'
    rmse_data = pd.read_csv(rmse_path)

    # Process prop_name
    rmse_data['prop_name_all'] = rmse_data['prop_name'].apply(lambda x: x.split('/')[2])
    rmse_data['ss'] = rmse_data['prop_name'].apply(lambda x: x.split('/')[0])
    # print median by proposal and prop_name
    for prop in rmse_data['prop_name_all'].unique():
        # get median avg_rmse by prop_name_all
        median = rmse_data.loc[rmse_data['prop_name_all'] == prop, 'avg_rmse_rc'].median()
        # get median ss corresponding to the median
        ss = rmse_data.loc[rmse_data['avg_rmse_rc'] == median, 'ss'].values[0]
        print(f'{model} {prop} {ss} median: {median}')
    rmse_data['prop_name_all'] = rmse_data['prop_name_all'].replace({'rw': 'RW', 'first_order': 'FO', 'second_order': 'SO'})
    rmse_data['prop_name_all'] = pd.Categorical(rmse_data['prop_name_all'], categories=['RW', 'FO', 'SO'], ordered=True)

    # Prepare data for boxplot
    data = [rmse_data.loc[rmse_data['prop_name_all'] == prop, 'avg_rmse_rc'].values for prop in ['RW', 'FO', 'SO']]

    

    # Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)

    # Style boxes: grey fill, black edges
    for box in bp['boxes']:
        box.set(facecolor='lightgray', edgecolor='black', linewidth=2)

    # Style other elements
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    # Set labels and titles
    ax.set_xticklabels(['RW', 'FO', 'SO'])
    if model.split('_')[0] == 'lgssm':
        ax.set_title('LGSS')
    else:
        ax.set_title(f'{model.split("_")[0].upper()}')
    ax.set_xlabel('Proposal')
    ax.set_ylabel('log-RMSE')
    ax.tick_params(axis='x', rotation=45)
    # log scale
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'outputs/{model}/boxplot_{model}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Run for all models
for model in models:
    boxplot_rmse(model)
