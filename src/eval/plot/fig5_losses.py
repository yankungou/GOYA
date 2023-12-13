import pandas as pd
import plotly.express as px
import argparse
import pprint
import os


parser = argparse.ArgumentParser(description='Loss comparison')

parser.add_argument('--ori', action='store_true', help='plot with results in the paper')
parser.add_argument('--save_dir', default='evaluation/figure', type=str, help='directory to save the figures')


def read_results(path):
    df = pd.read_csv(path)

    # ['Triplet, classifier(-)', 'Triplet, classifier(+)', 'NTXent, classifier(-)', 'NTXent, classifier(+)', 'Contrastive, classifier(-)', 'Contrastive, classifier(+)']
    custom_order = ['model_triplet_rm_clf', 'model_triplet', 'model_ntxent_rm_clf', 'model_ntxent', 'model_contrastive_rm_clf', 'GOYA']
    df = df[df['model'].isin(custom_order)]
    df['model'] = pd.Categorical(df['model'], categories=custom_order, ordered=True)
    df = df.sort_values(by='model')

    return df


def plot_fig5(genre_acc_list, style_acc_list, dc_list, save_path):
    mul_acc_list = [0.01 * x * 0.01 * y for x, y in zip(genre_acc_list, style_acc_list)]

    model_list = ['Triplet, classifier(-)', 'Triplet, classifier(+)', 'NTXent, classifier(-)', 'NTXent, classifier(+)', 'Contrastive, classifier(-)', 'Contrastive, classifier(+)']
    color_list = ['Triplet', 'Triplet', 'NTXent', 'NTXent', 'Contrastive', 'Contrastive']
    symbol_list = ['', 'Classifier', '', 'Classifier', '', 'Classifier']

    x_label = r'$\text{Acc (Genre} \times \text{Style)}$'
    y_label = r'$\text{DC (lower better)}$'
    df_mul = pd.DataFrame(list(zip(mul_acc_list, dc_list, model_list, color_list, symbol_list)),
                columns =[x_label, y_label, 'With different losses', 'Loss type', ' '])

    # template: plotly_white or simple_white or none
    fig_mul = px.scatter(df_mul, x=x_label, y=y_label, color="Loss type", symbol=" ", template='plotly_white', 
                        trendline="ols", trendline_scope="overall", 
                        trendline_color_override="purple", 
    )
    # fig_mul.update_traces(marker_size=10)
    fig_mul['data'][-1]['showlegend']=False

    fig_mul.update_traces(marker=dict(size=15,
                                    line=dict(width=2,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
    fig_mul.update_layout(legend=dict(
        yanchor="top",
        y=1,
        xanchor="left",
        x=0.01,
        font=dict(size = 15),
        title=None,
        ),
        xaxis=dict(
            titlefont_size=22,
            tickfont_size=18,
            showline=True, 
            linewidth=1, 
            linecolor='black'
        ),
        yaxis=dict(
            titlefont_size=22,
            tickfont_size=18,
            showline=True, 
            linewidth=1, 
            linecolor='black'
        ),
    )

    fig_mul.write_image(save_path)
    # results = px.get_trendline_results(fig_mul)
    # print(results)
    # results.query("0").px_fit_results.summary()
    print(f'fig5 is saved to {save_path}')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    pprint.pprint(args.__dict__, indent=2)
    
    os.makedirs(args.save_dir, exist_ok=True)

    if args.ori:
        genre_acc_list = [74.32, 74.13, 75.02, 75.93, 69.08, 69.70]
        style_acc_list = [39.34, 39.59, 52.40, 47.49, 46.79, 50.90]
        dc_list = [0.359, 0.366, 0.472, 0.413, 0.364, 0.367]
        save_path = os.path.join(args.save_dir, 'fig5_ori_results.pdf')
    else:
        clf_results_path = 'evaluation/results/clf_results.txt'
        df_clf = read_results(clf_results_path)
        genre_acc_list = df_clf['content'].tolist()
        style_acc_list = df_clf['style'].tolist()

        dc_results_path = 'evaluation/results/dc_results.txt'
        df_dc = read_results(dc_results_path)
        dc_list = df_dc['DC'].tolist()
        save_path = os.path.join(args.save_dir, 'fig5_replication.pdf')

    plot_fig5(genre_acc_list, style_acc_list, dc_list, save_path)