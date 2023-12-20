import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import argparse
import pprint
import os


parser = argparse.ArgumentParser(description='Disentanglement and classification evaluation with different embedding sizes when only one single layer is set in the content and style encoder.')

parser.add_argument('--ori', action='store_true', help='plot with results in the paper')
parser.add_argument('--save_dir', default='evaluation/figure', type=str, help='directory to save the figures')


def read_results(path):
    df = pd.read_csv(path)
    # [256, 512, 1024, 2048]
    custom_order = ['model_single_256', 'model_single_512', 'model_single_1024', 'model_single_2048']
    df = df[df['model'].isin(custom_order)]
    df['model'] = pd.Categorical(df['model'], categories=custom_order, ordered=True)
    df = df.sort_values(by='model')

    return df


def plot_fig6(genre_acc_list, style_acc_list, dc_list, save_path):
    colors = px.colors.qualitative.Bold
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    emb_size_list = [256, 512, 1024, 2048]

    fig.add_trace(go.Scatter(x=emb_size_list, y=genre_acc_list, name='Genre acc', line = dict(color=colors[0]), mode="lines+markers",
                            legendgroup="acc",
                            ), secondary_y=False)
    fig.add_trace(go.Scatter(x=emb_size_list, y=style_acc_list, name='Style acc', line = dict(color=colors[1]), mode="lines+markers",
                            legendgroup="acc",
                            ), secondary_y=False)
    fig.update_traces(marker_size=7)

    fig.add_trace(go.Scatter(x=emb_size_list, y=dc_list, name='DC',
                            marker=dict(
                                symbol="arrow",
                                angleref="previous",
                                size=13,
                                ),
                            mode="lines+markers",
                            line=dict(dash='dot', color='rgb(255,127,0)'),
                            legendgroup="dc",
                            ), secondary_y=True
    )

    # Edit the layout
    fig.update_layout(legend=dict(
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0.01,
        font=dict(size = 15),
        title=None,
        bgcolor='rgba(0,0,0,0)',
        ),
        template='simple_white',
        xaxis=dict(
            title=r'$\text{Embedding size}$',
            titlefont_size=22,
            tickfont_size=18,
            color='rgb(36,36,36)',
            showline=True, 
            linewidth=1, 
            linecolor='black'
            ),
        yaxis=dict(
            title=r'$\text{Acc (}\%\text{)}$',
            side="left",
            showline=True, 
            linewidth=1, 
            linecolor='black',
            titlefont_size=22,
            tickfont_size=18,
            color='rgb(36,36,36)'
            ),
        yaxis2=dict(
            title=r"$\text{DC (lower better)}$",
            side="right",
            showline=True, 
            linewidth=1, 
            linecolor='rgb(255,127,0)',
            titlefont=dict(
                color="rgb(255,127,0)"
            ),
            tickfont=dict(
                color='rgb(255,127,0)'
            ),
            range=[0.73, 0.83],
            titlefont_size=22,
            tickfont_size=18,
        ),
    )

    fig.write_image(save_path)
    print(f'fig6 is saved to {save_path}.')


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    pprint.pprint(args.__dict__, indent=2)
    
    os.makedirs(args.save_dir, exist_ok=True)

    if args.ori:
        genre_acc_list = [70.64, 75.02, 76.69, 77.10]
        style_acc_list = [52.06, 54.43, 55.56, 57.91]
        dc_list = [0.750, 0.808, 0.815, 0.814]
        save_path = os.path.join(args.save_dir, 'fig6_ori_results.pdf')
    else:
        clf_results_path = 'evaluation/results/clf_results.txt'
        df_clf = read_results(clf_results_path)
        genre_acc_list = df_clf['content'].tolist()
        style_acc_list = df_clf['style'].tolist()

        dc_results_path = 'evaluation/results/dc_results.txt'
        df_dc = read_results(dc_results_path)
        dc_list = df_dc['DC'].tolist()
        
        save_path = os.path.join(args.save_dir, 'fig6_replication.pdf')

    plot_fig6(genre_acc_list, style_acc_list, dc_list, save_path)