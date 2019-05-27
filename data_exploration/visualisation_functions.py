import pandas as pd
import numpy as np
from math import pi

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, NumeralTickFormatter, \
    FactorRange, Legend
from bokeh.layouts import gridplot
from bokeh.transform import jitter

from bokeh.io import export_svgs


def calculate_distribution_as_df(df: pd.DataFrame, col_name: str, is_categorical: str, bins: int):
    """
    Returns dataframe with data for corresponding distribution
    :param df: dataframe with data
    :param col_name: indicates the column needed to calculate distribution
    :param is_categorical: whether distribution should be for categorical or numerical variables
    :param bins: number of bins in histogram
    :return: distribution as dataframe
    """
    if is_categorical:
        col_counts = df[col_name].value_counts(dropna=False, normalize=True)
        col_df = pd.DataFrame(col_counts)

    else:
        col_counts = df[col_name].value_counts(dropna=False, normalize=True, bins=bins)
        col_df = pd.DataFrame(col_counts)
        null_val_count = 1 - col_df[col_name].sum()
        null_val_df = pd.DataFrame({col_name: [null_val_count]}, index=["Nan"])
        col_df = col_df.append(null_val_df)

    col_df = col_df.reset_index().rename(columns={'index': 'category'})
    col_df.category = col_df.category.apply(str)

    return col_df


def plot_bar_chart_distribution(df: pd.DataFrame, col_name: str, is_categorical: bool = True, plot_width: int = 330,
                                plot_height: int = 330, colour: str = '#00BFA5', bins: int = 10):
    """
    Creates figure with distribution as bar chart
    :param df: dataframe with data
    :param col_name: indicates the column needed to to create distribution figure
    :param is_categorical: indicates if plot uses categorical or numerical variables
    :param plot_width: figure's width (default = 330)
    :param plot_height: figure's height (default = 330)
    :param colour: fill colour for bar chart
    :param bins: number of bins
    :return: figure with  bar chart distribution
    """

    distribution_df = calculate_distribution_as_df(df, col_name, is_categorical, bins=bins)

    source = ColumnDataSource(distribution_df)

    hover = HoverTool()

    p = figure(x_range=source.data["category"].tolist(), title=col_name, plot_width=plot_width,
               plot_height=plot_height, tools=["save", hover])

    p.title.text_font = p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = "Helvetica Neue"
    p.xgrid.visible = p.ygrid.visible = False
    p.xaxis.major_label_orientation = pi / 4
    p.yaxis[0].formatter = NumeralTickFormatter(format='0%')

    p.vbar(x="category", width=0.9, top=col_name, source=source, color=colour)

    tooltips = [("variable", "@category"),
                (col_name, "@" + col_name + '{0%}')]

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = tooltips

    return p


def plot_multiple_bar_chart_distribution(df: pd.DataFrame, col_name: str, group_category: str,
                                         vars_to_drop: list = [], is_categorical: bool = True, plot_width: int = 330,
                                         plot_height: int = 330, colours: list = ["#8c9eff", "#536dfe"]):
    """

    :param df:
    :param col_name:
    :param group_category:
    :param vars_to_drop:
    :param is_categorical:
    :param plot_width:
    :param plot_height:
    :param colours:
    :return:
    """

    grouped_table = df.groupby([group_category, col_name]).size() / df.groupby([group_category]).size()

    if len(vars_to_drop) > 0:
        grouped_table = grouped_table.drop(vars_to_drop)

    grouped_df = pd.DataFrame(grouped_table)
    grouped_df = grouped_df.rename(columns={0: "category"}).reset_index()

    index_tuple = [(col_cat, group_cat) for col_cat in grouped_df[col_name].unique()
                   for group_cat in grouped_df[group_category].unique()]

    percentages = []

    for tuple_ in index_tuple:
        col_cat = tuple_[0]
        group_cat = tuple_[1]
        per = grouped_df[(grouped_df[group_category] == group_cat) & (grouped_df[col_name] == col_cat)]["category"]
        percentages.append(per)
    percentages = tuple(percentages)

    colour_combinations = int(len(index_tuple) / 2)
    fill_colours = colours * colour_combinations

    source = ColumnDataSource(data=dict(x=index_tuple, percentages=percentages, fill_colours=fill_colours))

    p = figure(x_range=FactorRange(*index_tuple), plot_height=plot_height, plot_width=plot_width,
               title="Distribution of " + col_name + " by " + group_category, toolbar_location=None,
               tools="save, hover")

    p.vbar(x='x', top='percentages', color='fill_colours', width=0.9, source=source)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    p.yaxis[0].formatter = NumeralTickFormatter(format='0%')

    tooltips = [("variable", "@x"),
                ('percentage', "@percentages" + '{0%}')]

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = tooltips

    return p


def plot_multiple_correlations(df: pd.DataFrame, col1: str, col2: str, col_category: str, min_corr: float,
                               plot_width: int = 500, plot_height: int = 400, jitter_scale: float = 0.4,
                               circle_colours: list = ['#ffd54f', '#7986cb', '#4db6ac', '#f06292'],
                               line_colours: list = ['#ffb300', '#3f51b5', '#00897b', '#d81b60']):
    """
    Plots chart with correlation lines and scatter plots by category
    :param df: dataframe with data
    :param col1: column name for first numerical variable
    :param col2: column name for second numerical variable
    :param col_category: column category
    :param min_corr: minimum correlation required for one of both categories to plot the chart
    :param plot_width: plot's width
    :param plot_height: plot's height
    :param jitter_scale: jitter space separation for scatter plot
    :param circle_colours: circle colours for scatter plot
    :param line_colours: line colours for correlation lines
    :return: chart with correlation lines and scatter plots by category
    """
    legend_it = []

    p = figure(title='Correlation between {} and {} by {}'.format(col1, col2, col_category), plot_width=plot_width,
               plot_height=plot_height, tools=["save"])

    correlations = []

    for ind, cat in enumerate(df[col_category].dropna().unique()):
        df_for_col = df[df[col_category] == cat].dropna()

        correlations.append(df_for_col[col1].corr(df_for_col[col2]))

    if any(corr >= min_corr for corr in correlations):

        for ind, cat in enumerate(df[col_category].dropna().unique()):
            df_for_col = df[df[col_category] == cat].dropna()
            source = ColumnDataSource(df_for_col)
            c = p.circle(x=jitter(col1, width=jitter_scale, range=p.x_range, distribution='normal'), y=col2,
                         color=circle_colours[ind], alpha=1 - (ind * 0.5 ** (ind)), source=source)

            legend_it.append(('{} (r = {})'.format(cat, round(correlations[ind], 2)), [c]))

        for ind, cat in enumerate(df[col_category].dropna().unique()):
            df_for_col = df[df[col_category] == cat].dropna()
            par = np.polyfit(df_for_col[col1], df_for_col[col2], 1, full=True)
            slope = par[0][0]
            intercept = par[0][1]
            y_predicted = [slope * i + intercept for i in df_for_col[col1]]

            df_for_col['y_predicted'] = y_predicted

            source = ColumnDataSource(df_for_col)

            p.line(x=col1, y='y_predicted', color=line_colours[ind], alpha=0.8, line_width=2, source=source)

        p.title.text_font = p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = "Helvetica Neue"
        p.xgrid.visible = p.ygrid.visible = False

        p.xaxis.axis_label = col1
        p.yaxis.axis_label = col2

        legend = Legend(items=legend_it, location=(0, 0))
        p.add_layout(legend, 'below')

        return p, True

    else:
        return -1, False


def create_plot_layout(df: pd.DataFrame, number_columns: int, plot_func: callable, ignore_cols: list = [], **kwargs):
    """
    Plots a grid of plots based on number of columns given and plot function to be used
    :param df: dataframe with all columns to be plotted
    :param number_columns: the number of columns that the grid should have
    :param plot_func: the plot function to call for each plot in the grid
    :param ignore_cols: columns that should not be used for plotting
    :param kwargs: any extra variable that the plot function may require
    :return: plots a grid with graphs
    """
    layout_grid = []
    count = 0
    row = []
    cols_to_plot = [col for col in df.columns if col not in ignore_cols]

    for col in cols_to_plot:
        p = plot_func(df, col, **kwargs)
        row.append(p)
        count = count + 1
        if (count == number_columns) or (col == df.columns[len(df.columns) - 1]):
            layout_grid.append(row)
            row = []
            count = 0
    layout = gridplot(layout_grid)
    show(layout)


def create_corr_plot_layout(df: pd.DataFrame, number_columns: int, plot_func: callable, min_corr: float = 0.3,
                            ignore_cols: list = [], **kwargs):
    """
    Plots a grid of corr plots based on number of columns given and plot function to be used
    :param df: dataframe with all columns to be plotted
    :param number_columns: the number of columns that the grid should have
    :param plot_func: the plot function to call for each plot in the grid
    :param min_corr:
    :param ignore_cols: columns that should not be used for plotting
    :param kwargs: any extra variable that the plot function may require
    :return: plots a grid with graphs
    """

    cols_to_plot = [col for col in df.columns if col not in ignore_cols]

    for ind, col1 in enumerate(cols_to_plot):
        second_cols = cols_to_plot[ind + 1:]
        if len(second_cols) == 0:
            break;
        layout_grid = []
        count = 0
        row = []
        for col2 in second_cols:
            p, one_min_corr = plot_func(df, col1, col2, min_corr=min_corr, **kwargs)
            if one_min_corr:
                row.append(p)
                count = count + 1
                if (count == number_columns) or (col1 == df.columns[len(df.columns) - 1]):
                    layout_grid.append(row)
                    row = []
                    count = 0
        if len(layout_grid) != 0:
            layout = gridplot(layout_grid)
            show(layout)


def plot_roc_curve(categories, title: str = 'ROC curve', x_axis_label: str = 'False Positive Rate (FPR)',
                   y_axis_label: str = 'True Positive Rate (TPR)', line_width=2, plot_width=450, plot_height=450):
    """
    Plots ROC curve
    :param fpr: list of false positve rates
    :param tpr: list of true positive rates
    :param title: title for the plot (default: ROC curve)
    :param x_axis_label: name for the x axis (default: False Positive Rate (FPR))
    :param y_axis_label: name for the y axis (default: True Positive Rate (TPR))
    :return: plots ROC curve plot figure
    """

    p = figure(plot_width=plot_width, plot_height=plot_height, title=title)

    for category, values in categories.items():

        p.line(values['fpr'], values['tpr'], line_width=line_width, line_color=values['line_colour'],
               line_dash=values['line_dash'], alpha=values['alpha'], legend=category)

    mid_line = np.arange(0, 1.1, 0.1)
    p.line(mid_line, mid_line, line_color="#616161", line_dash='dashed')

    p.legend.location = "bottom_right"

    p.y_range.start = 0
    p.x_range.start = 0
    p.y_range.end = 1
    p.x_range.end = 1

    p.xaxis.axis_label = x_axis_label
    p.yaxis.axis_label = y_axis_label
    p.title.text_font = p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = "Helvetica Neue"

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p


def save_plot(p, file_name, path='../static/images/'):
    """
    Saves Bokeh plot figure as svg
    :param p: Bokeh plot figure
    :param file_name: name for the plot
    :param path: path where file is saved
    """
    p.output_backend = "svg"
    export_svgs(p, filename=path + file_name + '.svg')
