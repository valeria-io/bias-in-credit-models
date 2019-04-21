import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, isnan

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, PrintfTickFormatter, BasicTickFormatter, NumeralTickFormatter, \
    FactorRange
from bokeh.layouts import gridplot
from bokeh.themes import Theme
import seaborn as sns


def calculate_distribution_as_df(df, col_name, is_categorical):
    """
    Returns dataframe with data for corresponding distribution
    :param df: dataframe with data
    :param col_name: indicates the column needed to calculate distribution
    :param is_categorical: whether distribution should be for categorical or numerical variables
    :return: distribution as dataframe
    """
    if is_categorical:
        col_counts = df[col_name].value_counts(dropna=False, normalize=True)
        col_df = pd.DataFrame(col_counts)

    else:
        col_counts = df[col_name].value_counts(dropna=False, normalize=True, bins=10)
        col_df = pd.DataFrame(col_counts)
        null_val_count = 1 - col_df[col_name].sum()
        null_val_df = pd.DataFrame({col_name: [null_val_count]}, index=["Nan"])
        col_df = col_df.append(null_val_df)

    col_df = col_df.reset_index().rename(columns={'index': 'category'})
    col_df.category = col_df.category.apply(str)

    return col_df


def plot_bar_chart_distribution(df, col_name, is_categorical=True, plot_width=330, plot_height=330, colour='#00BFA5'):
    """
    Creates figure with distribution as bar chart
    :param df: dataframe with data
    :param col_name: indicates the column needed to to create distribution figure
    :param is_categorical: indicates if plot uses categorical or numerical variables
    :param plot_width: figure's width (default = 330)
    :param plot_height: figure's height (default = 330)
    :param colour: fill colour for bar chart
    :return: figure with  bar chart distribution
    """

    distribution_df = calculate_distribution_as_df(df, col_name, is_categorical)

    source = ColumnDataSource(distribution_df)

    hover = HoverTool()

    p = figure(x_range=source.data["category"].tolist(), title=col_name, plot_width=plot_width,
               plot_height=plot_height, tools=["save", hover])

    p.title.text_font = p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = "Helvetica Neue"
    p.xgrid.visible = p.ygrid.visible = False
    p.xaxis.major_label_orientation = pi / 4
    p.yaxis[0].formatter = NumeralTickFormatter(format='0 %')

    p.vbar(x="category", width=0.9, top=col_name, source=source, color=colour)

    tooltips = [("variable", "@category"),
                (col_name, "@y" + col_name)]

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = tooltips

    return p


def create_plot_layout(df, number_columns, plot_func, **kwargs):
    """
    Plots a grid of plots based on number of columns given and plot function to be used
    :param df: dataframe with all columns to be plotted
    :param number_columns: the number of columns that the grid should have
    :param plot_func: the plot function to call for each plot in the grid
    :param kwargs: any extra variable that the plot function may require
    :return: plots a grid with graphs
    """
    layout_grid = []
    count = 0
    row = []
    for col in df.columns:
        p = plot_func(df, col, **kwargs)
        row.append(p)
        count = count + 1
        if (count == number_columns) or (col == df.columns[len(df.columns) - 1]):
            layout_grid.append(row)
            row = []
            count = 0
    layout = gridplot(layout_grid)
    show(layout)