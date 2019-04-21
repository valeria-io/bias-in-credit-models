import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, isnan

from bokeh.plotting import figure, show
from bokeh.io import output_notebook, curdoc
from bokeh.models import HoverTool, ColumnDataSource, PrintfTickFormatter, BasicTickFormatter, NumeralTickFormatter, \
    FactorRange
from bokeh.layouts import column, gridplot
from bokeh.themes import Theme
import seaborn as sns


def plot_bar_chart(df, x_col_name, y_col_name, plot_width=330, plot_height=330, colour='#00BFA5', **kwargs):
    source = ColumnDataSource(df)

    hover = HoverTool()

    p = figure(x_range=source.data[x_col_name].tolist(), title=y_col_name, plot_width=plot_width,
               plot_height=plot_height, tools=["save", hover])
    # fonts
    p.title.text_font = p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = "Helvetica Neue"
    # line visibility
    p.xgrid.visible = p.ygrid.visible = False
    # axis orientation
    p.xaxis.major_label_orientation = pi / 4
    # Percentage formatting
    p.yaxis[0].formatter = NumeralTickFormatter(format='0 %')
    # bar chart settings
    p.vbar(x=x_col_name, width=0.9, top=y_col_name, source=source, color=colour)

    # hover settings
    tooltips = [(x_col_name, "@" + x_col_name),
                (y_col_name, "@y" + y_col_name + kwargs.get('y_tooltip_format', ''))]

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = tooltips

    return p
