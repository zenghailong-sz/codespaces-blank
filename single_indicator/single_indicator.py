import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from matplotlib.axes import Axes
from adjustText import adjust_text
import re


# todo 图例展示，加两种：一、单独单折线图；二、单独单柱状图
# todo 加一个全局单函数，图例展示，包括两组单独单数据源， 分别用柱状图展示和折现图展示。提供参数是否将指标数值画到图上展示。


plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['figure.dpi'] = 300
plt.rcParams["legend.fontsize"] = 6
plt.rcParams['axes.titlesize'] = 6
plt.rcParams['figure.figsize'] = (8, 4)


def draw_graph(data, bars, lines, title, is_show_text, is_legend_out, ax):
    font_size = plt.rcParams['axes.titlesize']
    # if not ax:
    #     ax_new = plt.subplots()
    # else:
    #     ax_new = ax
    ax_new = ax if ax else plt.subplots()[1]
    if bars:
        ax_bar = data.plot.bar(y=bars['columns'], use_index=True, rot=0, ax=ax_new)
    if lines:
        # ax_lines_list = [
        #     data.plot(y=column, use_index=True, rot=0, secondary_y=True, ax=ax_new, alpha=0.5)
        #     if bars else data.plot(y=column, use_index=True, rot=0, ax=ax_new, alpha=0.5) for
        #     column in lines['columns']]
        ax_lines_list = [
            data.plot(y=column, use_index=True, rot=0, secondary_y=True if bars else False, ax=ax_new, alpha=0.5) for
            column in lines['columns']]

    left_ax = ax_bar if bars else ax_lines_list[0]
    left_ax.set_xlabel('')
    left_ax.set_title(title)
    left_ax.tick_params(labelsize=font_size)

    left_format = bars['format'] if bars else lines['format']
    left_ax.yaxis.set_major_formatter(left_format)

    right_ax = ax_lines_list[0] if bars and lines else None
    if right_ax:
        right_ax.tick_params(labelsize=font_size)
        right_ax.yaxis.set_major_formatter(lines['format'])

    if is_show_text:
        texts = []
        if bars:
            for contain_bar in ax_bar.containers:
                labels = ['' if v == 0 else f'{round(v, bars["precision"])}{bars["unit"]}' for v in
                          contain_bar.datavalues]
                texts += ax_bar.bar_label(contain_bar, labels=labels, fontsize=font_size)
        if lines:
            for column_line, ax_line in zip(lines['columns'], ax_lines_list):
                for index, value in enumerate(data.iloc[:, column_line]):
                    if pd.isna(value):
                        continue
                    texts.append(
                        ax_line.text(index, value, str(round(value, lines['precision'])) + lines['unit'], ha='center', va='bottom',
                                     fontsize=font_size))
        adjust_text(texts, autoalign='y', ax=ax, only_move={'points': 'y', 'text': 'y', 'objects': 'y'})

    if is_legend_out:
        if bars:
            ax_bar.legend(loc='lower left', borderaxespad=0., bbox_to_anchor=(0, 1.01))
        if lines:
            for ax_line in ax_lines_list:
                ax_line.legend(loc='lower right', borderaxespad=0., bbox_to_anchor=(1, 1.01))


class SingleIndicator:
    """
    单一指标类: 索引列数据格式，月度数据为：2020M01...2020M12, 季度数据为：2020Q1..2020Q4，年度数据为2020
    """

    # 季度原始数据， 索引：季度数据（如2019q1,2019q2)，月度数据（如2019M0，2019M12），年度数据（如2012年，2013年）, 唯一的列为具体指标数据列
    _df: pd.DataFrame
    # 新的数据包含：索引、指标数据列、环比增长额、环比增长率、同比增长额、同比增长率
    _df_new: pd.DataFrame
    # 年度数据包含：年度索引、指标数据列、环比增长额、环比增长率
    _df_year: pd.DataFrame
    # 数据单位: 比如，十亿，亿，千万，百万，十万，万，千等等
    _unit: str
    # 指标数据是否过去12个月累计
    _is_accumulative: bool
    # 指标名称
    _name: str
    # 数据代表几个月的数据: 1,3,12，分别代表是月数据、季度数据、年度数据
    _num_months: int

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def fd_new(self):
        return self._df_new

    @property
    def fd_year(self):
        return self._df_year

    @property
    def is_accumulative(self):
        return self._is_accumulative

    @property
    def period(self):
        return {
            12: "年度",
            3: "季度",
            1: "月度"
        }[self._num_months]

    def _extract_unit_and_name(self):  # optimized by chatgpt
        column0_name = self._df.columns[0]
        unit_match = re.search(r'\((.*?)\)', column0_name)
        unit = unit_match.group(1) if unit_match else ''
        name = column0_name.split('(')[0].strip()
        return unit, name

    def __init__(self, is_accumulative: bool, num_months: int, **kwargs):
        """
        构造函数
        :param is_accumulative: 数据是过去12个月的累计值, num_moths==12, 则默认就是True
        :param num_months: 1: 月数据， 3：季数据，12：年数据
        可选的形式参数： csv: str，value: Series(包含有索引和单列值)
        """

        if 'csv' in kwargs:
            self._df = pd.read_csv(kwargs['csv'], index_col=0)
        elif 'value' in kwargs:
            self._df = pd.DataFrame(kwargs['value'])
        else:
            raise Exception('非法参数！')

        self._num_months = num_months
        self._is_accumulative = is_accumulative
        self._unit, self._name = self._extract_unit_and_name()
        self._init_table()

    def _init_table(self):
        """
            初始化两个表格，一个是_df_new, 一个是_df_year
        """

        def drop_head_tail(dfa: pd.DataFrame, period: str):  # chatgpt version
            if period not in ['4', '12']:
                raise Exception('参数错误！')

            start_idx = dfa.index.get_loc(dfa.index[dfa.index.str.endswith('1')].min())
            end_idx = dfa.index.get_loc(dfa.index[dfa.index.str.endswith(period)].max())
            return dfa.iloc[start_idx:end_idx + 1]

        self._df_new = self._df.copy()
        self._df_new['环比增加额({})'.format(self._unit)] = self._df_new[self._df_new.columns[0]].diff(periods=1)
        self._df_new['环比增长率(%)'] = self._df_new[self._df_new.columns[0]].pct_change(periods=1) * np.sign(
            self._df_new[self._df_new.columns[0]].shift(periods=1))
        self._df_new['环比增长率(%)'] = self._df_new['环比增长率(%)'].apply(lambda x: x if pd.isna(x) else round(x * 100, 2))
        if self._num_months in [1, 3]:  # 月度和季度数据添加一个同比数据
            self._df_new['同比增加额({})'.format(self._unit)] = self._df_new[self._df_new.columns[0]].diff(
                periods=12 // self._num_months)
            self._df_new['同比增长率(%)'] = self._df_new[self._df_new.columns[0]].pct_change(
                periods=12 // self._num_months) * np.sign(
                self._df_new[self._df_new.columns[0]].shift(periods=12 // self._num_months))
            self._df_new['同比增长率(%)'] = self._df_new['同比增长率(%)'].apply(lambda x: x if pd.isna(x) else round(x * 100, 2))

        if self._num_months == 12:  # 原始数据就是年度数据
            self._df_year = self._df.copy()
        elif self._num_months == 3:  # 原始数据是季度数据
            self._df_year = self._df[
                self._df.index.str[-1] == '4'].copy() if self._is_accumulative else drop_head_tail(self._df,
                                                                                                   '4').groupby(
                drop_head_tail(self._df, '4').index.str[0:4]).sum()

        elif self._num_months == 1:  # 原始数据是月度数据
            self._df_year = self._df[
                self._df.index.str[-2:] == '12'].copy() if self._is_accumulative else drop_head_tail(self._df,
                                                                                                     '12').groupby(
                drop_head_tail(self._df, '12').index.str[0:4]).sum()

        self._df_year.rename(lambda x: str(x)[0:4] + '年', axis='index', inplace=True)
        self._df_year.index.rename('年度', inplace=True)
        self._df_year['环比增加额({})'.format(self._unit)] = self._df_year[self._df_year.columns[0]].diff(periods=1)
        self._df_year['环比增长率(%)'] = self._df_year[self._df_year.columns[0]].pct_change(periods=1) * np.sign(
            self._df_year[self._df_year.columns[0]].shift(periods=1))
        self._df_year['环比增长率(%)'] = self._df_year['环比增长率(%)'].apply(lambda x: x if pd.isna(x) else round(x * 100, 2))

    def show_graph(self, is_fiscal: bool = False, show_additional: bool = True, ax: Axes = None):  # chatgpt version
        df = self._df_year if is_fiscal else self._df_new

        columns_0 = [0]
        columns_1_3 = [0, 1] if is_fiscal or self._num_months == 12 else [0, 1, 3]
        columns_2_4 = [2, 4] if not is_fiscal and self._num_months in [1, 3] else [2]

        bars = {'columns': columns_0 if not show_additional else columns_1_3,
                'format': StrMethodFormatter('{x:1.0f}' + self._unit),
                'precision': 3, 'unit': self._unit}
        lines = {'columns': columns_2_4,
                 'format': PercentFormatter(decimals=0), 'precision': 2, 'unit': '%'}

        title = f'{self.name}趋势图{self.period}'
        is_show_text = True
        is_legend_out = True

        draw_graph(df, bars, lines, title, is_show_text, is_legend_out, ax)


def stacked_graph_show(single_indicators: list[SingleIndicator], is_year, title, ax, is_percentage):
    columns = [indicator.fd_year.iloc[:, 0] if is_year else indicator.fd_new.iloc[:, 0] for indicator in
               single_indicators]
    new_df = pd.concat(columns, axis=1, join='inner')

    if is_percentage:
        new_df['合计'] = new_df.sum(axis='columns')
        new_df = (new_df.iloc[:, :-1].div(new_df['合计'], axis='index') * 100).round(2)
        new_df.rename(columns={col: col.split('(')[0] + '占比(%)' for col in new_df.columns}, inplace=True)
    ax_stacked_bar = new_df.plot.bar(stacked=True, rot=0, ax=ax, title=title)

    texts_2d = [ax_stacked_bar.bar_label(contain_bar,
                                         labels=['' if v == 0 else f'{round(v, 3)}' for v in contain_bar.datavalues],
                                         fontsize=6, label_type='center') for contain_bar in ax_stacked_bar.containers]
    texts_1d = [x for sublist in texts_2d for x in sublist]
    texts_1d += ax_stacked_bar.bar_label(ax_stacked_bar.containers[-1], fontsize=6, label_type='edge')

    adjust_text(texts_1d, autoalign='y', ax=ax_stacked_bar, only_move={'points': 'y', 'text': 'y', 'objects': 'y'})
    ax_stacked_bar.set_xlabel('')
