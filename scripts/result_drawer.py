import pdb
import sys
import os
import json
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 10
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 10
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')
from scrabble.common import *
from scrabble.eval_func import *


import plotter
from plotter import save_fig

building_anon_map = {
    'ebu3b': 'A-1',
    'bml': 'A-2',
    'ap_m': 'A-3',
    'ghc': 'B-1',
    'sdh': 'C-1',
}
colors = ['firebrick', 'deepskyblue', 'darkgreen', 'goldenrod']
inferencer_names = ['zodiac', 'al_hong', 'scrabble']
LINESTYLES = [':', '--', '-.', '-']
FIG_DIR = './figs'

def average_data(xs, ys, target_x):
    target_y = np.zeros((1, len(target_x)))
    for x, y in zip(xs, ys):
        yinterp = np.interp(target_x, x, y)
        target_y += yinterp / len(ys) * 100
    return target_y.tolist()[0]


def plot_scrabble():
    buildings = ['ebu3b', 'uva_cse', 'sdh', 'ghc']
    #buildings = ['sdh', 'ebu3b']
    outputfile = FIG_DIR + '/pointonly_notransfer.pdf'

    fig, axes = plt.subplots(1, len(buildings))
    xticks = [0, 10] + list(range(50, 251, 50))
    xticks_labels = [''] + [str(n) for n in xticks[1:]]
    yticks = range(0,101,20)
    yticks_labels = [str(n) for n in yticks]
    xlim = (-5, xticks[-1]+5)
    ylim = (yticks[0]-2, yticks[-1]+5)
    interp_x = list(range(10, 250, 5))
    for ax_num, (ax, building) in enumerate(zip(axes, buildings)): # subfigure per building
        xlabel = '# of Examples'
        ylabel = 'Score (%)'
        title = building_anon_map[building]
        linestyles = deepcopy(LINESTYLES)
        for inferencer_name in inferencer_names:
            if building == 'uva_cse' and inferencer_name == 'scrabble':
                continue
            xs = []
            ys = []
            xss = []
            f1s = []
            mf1s = []
            for i in range(0, EXP_NUM):
                with open('result/pointonly_notransfer_{0}_{1}_{2}.json'
                          .format(inferencer_name, building, i)) as  fp:
                    data = json.load(fp)
                xss.append([datum['learning_srcids'] for datum in data])
                if inferencer_name == 'al_hong':
                    f1s.append([datum['metrics']['f1_micro'] for datum in data])
                    mf1s.append([datum['metrics']['f1_macro'] for datum in data])
                else:
                    f1s.append([datum['metrics']['f1'] for datum in data])
                    mf1s.append([datum['metrics']['macrof1'] for datum in data])
            xs = xss[0] # Assuming all xss are same.
            f1 = average_data(xss, f1s, interp_x)
            mf1 = average_data(xss, mf1s, interp_x)
            x = interp_x
            ys = [f1, mf1]
            if ax_num == 0:
                #data_labels = ['Baseline Acc w/o $B_s$',
                #               'Baseline M-$F_1$ w/o $B_s$']
                legends = ['MicroF1, {0}'.format(inferencer_name),
                           'MacroF1, {0}'.format(inferencer_name)
                           ]
            else:
                #data_labels = None
                legends = None

            _, plots = plotter.plot_multiple_2dline(
                x, ys, xlabel, ylabel, xticks, xticks_labels,
                yticks, yticks_labels, title, ax, fig, ylim, xlim, legends,
                linestyles=[linestyles.pop()]*len(ys), cs=colors)
    for ax in axes:
        ax.grid(True)
    for i in range(1,len(buildings)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(0,len(buildings)):
        if i != 1:
            axes[i].set_xlabel('')
    axes[0].legend(bbox_to_anchor=(3.2, 1.5), ncol=3, frameon=False)
    fig.set_size_inches((8,2))
    save_fig(fig, outputfile)


def plot_ir2tagsets(target_sources):
    fig, axes = plt.subplots(1, len(target_sources))
    outputfile = FIG_DIR + '/entity_iter.pdf'
    for ax, (target_building, source_building) in zip(axes, target_sources):
        plot_one_ir2tagsets(target_building, source_building, fig, ax)

    legend_ax = axes[0]
    #axes[0].legend(bbox_to_anchor=(2.2, 1.7), ncol=4, fontsize='small',
    #               frameon=False, columnspacing=0.5)
    legend_ax.legend(bbox_to_anchor=(0.1,0.96), ncol=4, fontsize='small',
              frameon=False, handletextpad=0.15, columnspacing=0.7)

    for i in range(1,len(target_sources)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(1,len(target_sources)):
        axes[i].set_xlabel('')
    axes[0].xaxis.set_label_coords(1.0, -0.2)
    plt.text(-0.09, 1.18, 'Accuracy: \nMacro$\mathrm{F_1}$: ', ha='center', va='center',
            transform=axes[0].transAxes, fontsize='small')

    fig.set_size_inches((4, 1.5))
    save_fig(fig, outputfile)


def plot_one_ir2tagsets(target_building, source_building,
                        fig=None, ax=None):
    title = '{0}$\mathrm{{\\Rightarrow}}${1}'.format(
        building_anon_map[source_building],
        building_anon_map[target_building]
    )
    linestyles = deepcopy(LINESTYLES)
    configs = get_ir2tagsets_configs(target_building, source_building)
    if target_building == 'ebu3b':
        configs.append({
            'use_brick_flag': True,
            'negative_flag': True,
            'source_building_list': [source_building, target_building],
            'target_building': target_building,
            'tagset_classifier_type': 'MLP',
            'task': 'ir2tagsets',
            'ts_flag': True,
        })
    xlabel = '# of Target Building Examples'
    ylabel = 'Score (%)'
    if not fig or not ax:
        fig, ax = plt.subplots(1, 1)
    for config in configs:
        filename = get_filename_for_ir2tagsets(target_building, config)
        with open(filename, 'r') as fp:
            res = json.load(fp)
        accuracy = res['accuracy']
        macrof1 = res['macrof1']
        xticks = [0, 10] + list(range(50, 201, 50))
        xticks_labels = [''] + [str(n) for n in xticks[1:]]
        yticks = range(0,101,20)
        yticks_labels = [str(n) for n in yticks]
        xlim = (50, xticks[-1])
        ylim = ((0, 100))
        #ylim = (yticks[0], yticks[-1])
        interp_x = list(range(10, 200, 5))
        ys = [accuracy, macrof1]
        if target_building == 'ebu3b':
            legends = [
                '{0},SA:{1}{2}'.format(
                #'Accruracy, {0},SA:{1}{2}'.format(
                    200 if len(config['source_building_list']) > 1 else 0,
                    'O' if config['use_brick_flag'] else 'X',
                    ',TS' if config['ts_flag'] else ''),
                '{0},SA:{1}{2}'.format(
                #'Macro-F1, {0},SA:{1}{2}'.format(
                    200 if len(config['source_building_list']) > 1 else 0,
                    'O' if config['use_brick_flag'] else 'X',
                    ',TS' if config['ts_flag'] else ''),
            ]
            #if inferencer_name == 'scrabble':
            #    legends.append('Accuracy, {0}'.format(inferencer_name))
        else:
            #data_labels = None
            legends = None


        _, plots = plotter.plot_multiple_2dline(
            interp_x, ys, xlabel, ylabel, xticks, xticks_labels,
            yticks, yticks_labels, None, ax, fig, ylim, xlim, legends,
            linestyles=[linestyles.pop()]*len(ys), cs=colors)
        ax.text(0.9, 0.15, title, transform=ax.transAxes, ha='right',
                backgroundcolor='white'
                )#, alpha=0)
    ax.grid(True)


def get_ir2tagsets_configs(target_building, source_building):
    configs = [
            {'use_brick_flag': True,
             'negative_flag': True,
             'source_building_list': [source_building, target_building],
             'target_building': target_building,
             'tagset_classifier_type': 'MLP',
             'task': 'ir2tagsets',
             'ts_flag': False,
             },
            {'use_brick_flag': True,
             'negative_flag': True,
             'source_building_list': [target_building],
             'target_building': target_building,
             'tagset_classifier_type': 'MLP',
             'task': 'ir2tagsets',
             'ts_flag': False,
             },
            #{'use_brick_flag': True,
            # 'negative_flag': True,
            # 'source_building_list': [target_building],
            # 'target_building': target_building,
            # 'tagset_classifier_type': 'StructuredCC',
            # 'task': 'ir2tagsets'
            # },
            {'use_brick_flag': False,
             'negative_flag': False,
             'source_building_list': [target_building],
             'target_building': target_building,
             'tagset_classifier_type': 'MLP',
             'task': 'ir2tagsets',
             'ts_flag': False,
             },
            ]
    return configs

def get_filename_for_ir2tagsets(target_building, config):
    filename = 'result/ir2tagsets_{target}_{source}_{sample_aug}_{ct}'\
            .format(
                    target = target_building,
                    source = config['source_building_list'][0],
                    sample_aug = 'sampleaug' if config['use_brick_flag']
                                 else 'noaug',
                    ct = config['tagset_classifier_type'].lower()
                    )
    if config.get('ts_flag', False):
        filename += '_ts'
    return filename

def calculate_ir2tagets_results(target_building,
                                source_building,
                                recalculate=False):
    EXP_NUM = 2
    default_configs = {
            'use_known_tags': True,
            'task': 'ir2tagsets',
            }
    configs = get_ir2tagsets_configs(target_building, source_building)
    if target_building == 'ebu3b':
        configs.append({
            'use_brick_flag': True,
            'negative_flag': True,
            'source_building_list': [source_building, target_building],
            'target_building': target_building,
            'tagset_classifier_type': 'MLP',
            'task': 'ir2tagsets',
            'ts_flag': True,
        })
    postfixes = [10, 11]


    for new_config in configs:
        fig, ax = plt.subplots(1, 1)
        interp_x = list(range(10, 200, 5))
        xss = []
        accs = []
        mf1s = []
        config = deepcopy(default_configs)
        config.update(new_config)
        for postfix in postfixes:
            config['postfix'] = str(postfix)
            try:
                res = query_result(config)
            except:
                pdb.set_trace()
                continue
            history = res['history']
            if not history:
                pdb.set_trace()
                continue
            x = []
            acc = []
            mf1 = []
            for hist in history:
                pred = hist['pred']
                target_srcids = [srcid for srcid in pred.keys()
                                 if (target_building == 'ebu3b' and
                                         srcid[0:3] in ['505', '506']) or
                                 (target_building == 'ghc' and
                                  '_' not in srcid)]
                pred = {srcid:pred[srcid] for srcid in target_srcids}
                truth = get_true_labels(target_srcids, 'tagsets')
                acc.append(get_accuracy(truth, pred))
                mf1.append(get_macro_f1(truth, pred))
                num_learning_srcids = hist['learning_srcids']
                if len(config['source_building_list']) == 2:
                    num_learning_srcids -= 200
                elif len(config['source_building_list']) > 2:
                    raise Exception('define this case')
                x.append(num_learning_srcids)
            mf1s.append(mf1)
            accs.append(acc)
            xss.append(x)
        averaged_acc = average_data(xss, accs, interp_x)
        averaged_mf1 = average_data(xss, mf1s, interp_x)
        res = {
                'accuracy': averaged_acc,
                'macrof1': averaged_mf1
                }
        filename = get_filename_for_ir2tagsets(target_building, new_config)
        with open(filename, 'w') as fp:
            json.dump(res, fp)





def crf_entity_result_dep():
    building_sets = [('ebu3b', 'ap_m'), ('ap_m', 'bml'),
                 ('ebu3b', 'ghc'), ('ghc', 'ebu3b'), ('ebu3b', 'bml', 'ap_m')] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    #building_sets = [('ebu3b', 'ghc'), ('ebu3b', 'ghc')]
    #building_sets = [('ap_m',), ('bml',),
    #             ('ghc',), ('ebu3b',), ('ap_m',)] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    fig, axes = plt.subplots(1, len(building_sets))
    with open('result/baseline.json', 'r') as fp:
        baseline_results = json.load(fp)

    cs = ['firebrick', 'deepskyblue']
    plot_list = list()
    acc_better_list = []
    mf1_better_list = []
    comp_xs = [10, 50, 150]
    for i, (ax, buildings) in enumerate(zip(axes, building_sets)):
        print(i)
        # Config
        ylim = (-2, 105)
        xlim = (-2, 205)

        # Baseline with source
        result = baseline_results[str(buildings)]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        baseline_acc = result['avg_acc']
        std_acc = result['std_acc']
        baseline_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Examples'
        ys = [baseline_acc, baseline_mf1]
        baseline_x = sample_numbers
        #xtick = sample_numbers
        #xtick_labels = [str(no) for no in sample_numbers]
        #xtick = [0] + [5] + xtick[1:]
        xtick = [10] + list(range(40, 205, 40))
        #xtick = list(range(0, 205, 40))
        xtick_labels = [str(n) for n in xtick]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        linestyles = [':', ':']
        if i == 2:
            data_labels = ['Baseline Acc w/ $B_s$',
                           'Baseline M-$F_1$ w/ $B_s$']
        else:
            data_labels = None
        title = anon_building_dict[buildings[0]]
        for building in  buildings[1:-1]:
            title += ',{0}'.format(anon_building_dict[building])
        title += '$\\Rightarrow${0}'.format(anon_building_dict[buildings[-1]])
        lw = 1.2
        _, plot = plotter.plot_multiple_2dline(baseline_x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        # Baseline without source
        result = baseline_results[str((list(buildings)[-1],))]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        avg_acc = result['avg_acc']
        std_acc = result['std_acc']
        avg_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Examples'
        ys = [avg_acc, avg_mf1]
        x = sample_numbers
        #xtick = sample_numbers
        #xtick_labels = [str(no) for no in sample_numbers]
        #xtick = list(range(0, 205, 40))
        #xtick_labels = [str(n) for n in xtick]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        linestyles = ['-.', '-.']
        if i == 2:
            data_labels = ['Baseline Acc w/o $B_s$',
                           'Baseline M-$F_1$ w/o $B_s$']
        else:
            data_labels = None
        title = anon_building_dict[buildings[0]]
        for building in  buildings[1:-1]:
            title += ',{0}'.format(anon_building_dict[building])
        title += '$\\Rightarrow${0}'.format(anon_building_dict[buildings[-1]])
        lw = 1.2
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        # Scrabble without source
        buildingfix = ''.join([buildings[-1]] * 2)
        filename = 'result/crf_entity_iter_{0}_char2tagset_iter_nosource1.json'\
                       .format(buildingfix)
        if not os.path.exists(filename):
            continue
        with open(filename, 'r') as fp:
            res = json.load(fp)
        source_num = 0
        srcid_lens = [len(r['learning_srcids']) - source_num for r in res]
        accuracy = [r['result']['entity']['accuracy'] * 100 for r in res]
        mf1s = [r['result']['entity']['macro_f1'] * 100 for r in res]
        x = srcid_lens
        ys = [accuracy, mf1s]
        linestyles = ['--', '--']
        if i == 2:
            data_labels = ['Scrabble Acc w/o $B_s$',
                           'Scrabble M-$F_1$ w/o $B_s$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        # Scrabble with source
        buildingfix = ''.join(list(buildings) + [buildings[-1]])

        filename_template = 'result/crf_entity_iter_{0}_char2tagset_iter_{1}.json'
        x = range(10, 205, 10)
        x_cands = []
        acc_cands = []
        mf1_cands = []
        for exp_num in range(0, 3):
            filename = filename_template.format(buildingfix, exp_num)
            if not os.path.exists(filename):
                continue
            with open(filename, 'r') as fp:
                res = json.load(fp)
            source_num = 200 * (len(buildings) - 1)
            x_cand = [len(r['learning_srcids']) - source_num for r in res]
            acc_cand = [r['result']['entity']['accuracy'] * 100 for r in res]
            mf1_cand = [r['result']['entity']['macro_f1'] * 100 for r in res]
            x_cands.append(x_cand)
            acc_cands.append(acc_cand)
            mf1_cands.append(mf1_cand)
        acc = lin_interpolated_avg(x, x_cands, acc_cands)
        mf1 = lin_interpolated_avg(x, x_cands, mf1_cands)
        ys = [acc, mf1]

        print(buildings)
        mf1_betters = []
        acc_betters = []
        for comp_x in comp_xs:
            try:
                comp_idx_target = x.index(comp_x)
                comp_idx_baseline = baseline_x.index(comp_x)
                acc_better = \
                    acc[comp_idx_target]/baseline_acc[comp_idx_baseline] - 1
                mf1_better = \
                    mf1[comp_idx_target]/baseline_mf1[comp_idx_baseline] - 1
                """
                acc_better = \
                    acc[comp_idx_target] - baseline_acc[comp_idx_baseline] - 1
                mf1_better = \
                    mf1[comp_idx_target] - baseline_mf1[comp_idx_baseline] - 1
                """
                mf1_betters.append(mf1_better)
                acc_betters.append(acc_better)
                print('srouce#: {0}'.format(comp_x))
                print('Acc\t baseline: {0}\t scrbl: {1}\t better: {2}\t'
                      .format(
                          baseline_acc[comp_idx_baseline],
                          acc[comp_idx_target],
                          acc_better
                          ))
                print('MF1\t baseline: {0}\t scrbl: {1}\t better: {2}\t'
                      .format(
                          baseline_mf1[comp_idx_baseline],
                          mf1[comp_idx_target],
                          mf1_better
                          ))
            except:
                pdb.set_trace()
        mf1_better_list.append(mf1_betters)
        acc_better_list.append(acc_betters)

        linestyles = ['-', '-']
        if i == 2:
            data_labels = ['Scrabble Acc w/ $B_s$',
                           'Scrabble M-$F_1$ w/ $B_s$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        if i == 2:
            ax.legend(bbox_to_anchor=(3.5, 1.53), ncol=4, frameon=False)
            #ax.legend(bbox_to_anchor=(3.2, 1.45), ncol=4, frameon=False)
    print('====================')
    print('Source nums: {0}'.format(comp_xs))
#    pdb.set_trace()
    mf1_better_avgs = [np.mean(list(map(itemgetter(i), mf1_better_list)))
                       for i, _ in enumerate(comp_xs)]
    acc_better_avgs = [np.mean(list(map(itemgetter(i), acc_better_list)))
                       for i, _ in enumerate(comp_xs)]
    print('MF1 better in average, {0}'.format(mf1_better_avgs))
    print('Acc better in average, {0}'.format(acc_better_avgs))


    fig.set_size_inches(9, 1.5)
    for ax in axes:
        ax.grid(True)
    for i in range(1,len(building_sets)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(0,len(building_sets)):
        if i != 2:
            axes[i].set_xlabel('')

    #legends_list = ['Baseline A', 'Baseline MF']
    #axes[2].legend(loc='best', legends_list)


    save_fig(fig, 'figs/crf_entity.pdf')
    subprocess.call('./send_figures')

if __name__ == '__main__':
    #calculate_ir2tagets_results('ghc', 'ap_m')
    #calculate_ir2tagets_results('ebu3b', 'ap_m')
    plot_ir2tagsets([('ebu3b', 'ap_m'), ('ghc', 'ap_m')])
