import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
import pdb

import argparse
import json
import subprocess
import pdb
from functools import reduce
import re
import numpy as np
import os

import plotter
from plotter import save_fig


anon_building_dict = {
    'ebu3b': 'A-1',
    'bml': 'A-2',
    'ap_m': 'A-3',
    'ghc': 'B-1'
}

def interpolate(x1, y1, x2, y2):
    return (x1 * y2 + x2 * y1) / (x1 + x2)


def lin_interpolated_avg(target_x, x_list, y_list):
    target_y = []
    for t_x in target_x:
        t_y_cands = []
        for given_x, given_y in zip(x_list, y_list):
            assert len(given_x) == len(given_y)
            left_x = -10000
            right_x = 10000
            left_y = None
            right_y = None
            for one_x, one_y in zip(given_x, given_y):
                if one_x <= t_x:
                    if one_x > left_x:
                        left_x = one_x
                        left_y = one_y
                if one_x >= t_x:
                    if one_x < right_x:
                        right_x = one_x
                        right_y = one_y
            if left_x == right_x and right_x == t_x:
                assert left_y == right_y
                t_y_cand = left_y
            else:
                t_y_cand = interpolate(t_x - left_x, left_y, 
                                       right_x - t_x, right_y)
            t_y_cands.append(t_y_cand)
        t_y = np.mean(t_y_cands)
        target_y.append(t_y)
    return target_y
            

def crf_result():
    #source_target_list = [('ebu3b', 'ap_m'), ('ebu3b', 'ap_m')]
    source_target_list = [('ebu3b', 'ap_m'), ('ghc', 'ebu3b')]
    #n_list_list = [#[(1000, 0), (1000,5), (1000,20), (1000,50), (1000,100), (1000, 150), (1000,200)],
    #               [(200, 0), (200,5), (200,20), (200,50), (200,100), (200, 150), (200,200)],
    #               [(0,5), (0,20), (0,50), (0,100), (0,150), (0,200)]]
    char_precs_list = list()
    phrase_f1s_list = list()
#fig, ax = plt.subplots(1, 1)
    fig, axes = plt.subplots(1,len(source_target_list))
    if isinstance(axes, Axes):
        axes = [axes]
    fig.set_size_inches(4, 1.5)
    cs = ['firebrick', 'deepskyblue']
    filename_template = 'result/crf_iter_{0}_char2ir_iter_{1}.json'
    n_s_list = [1000, 200, 0]

    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = ['--', '-.', '-']
        plot_list = list()
        legends_list = list()
        for n_s in n_s_list:
            if n_s == 0:
                buildingfix = ''.join([target, target])
            else:
                buildingfix = ''.join([source, target, target])
            n = n_s + 0
            filename = filename_template.format(buildingfix, n)
            if not os.path.exists(filename):
                continue
            with open(filename, 'r') as fp:
                data = json.load(fp)
            xs = [len(datum['learning_srcids']) - n_s for datum in data]
            f1s = []
            for datum in data:
                prec = datum['result']['crf']['phrase_precision'] * 100
                rec = datum['result']['crf']['phrase_recall'] * 100
                f1 = 2 * prec * rec / (prec + rec)
                f1s.append(f1)
            macrof1s = [datum['result']['crf']['phrase_macro_f1'] * 100 
                        for datum in data]
            ys = [f1s, macrof1s]
                #ys = [char_precs, phrase_f1s, char_macro_f1s, phrase_macro_f1s]
                #xlabel = '# of Target Building Samples'
            xlabel = None
            ylabel = 'Score (%)'
            xtick = list(range(0, 205, 40))
            #xtick = [0] + [5] + xtick[1:]
            xtick_labels = [str(n) for n in xtick]
            ytick = range(0,101,20)
            ytick_labels = [str(n) for n in ytick]
            xlim = (xtick[0]-2, xtick[-1]+5)
            ylim = (ytick[0]-2, ytick[-1]+5)
            if i == 0 or n_s == 1000:
                legends = [#'#S:{0}, Char Prec'.format(n_s),
                    '#$B_S$:{0}'.format(n_s),
    #'#S:{0}, Char MF1'.format(n_s),
                    '#$B_S$:{0}'.format(n_s),
                ]
            else:
                legends = None
    #legends_list += legends
            title = None
            _, plots = plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                             xtick_labels, ytick, ytick_labels, title, ax, fig, \
                             ylim, xlim, legends, xtickRotate=0, \
                             linestyles=[linestyles.pop()]*len(ys), cs=cs)
            text = '{0} $\\Rightarrow$ {1}'.format(\
                    anon_building_dict[source],
                    anon_building_dict[target])
            ax.text(0.8, 0.1, text, transform=ax.transAxes, ha='right',
                    backgroundcolor='white'
                    )#, alpha=0)
            plot_list += plots

    axes[0].legend(bbox_to_anchor=(0.15, 0.96), ncol=3, frameon=False)
    for ax in axes:
        ax.grid(True)
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    plt.text(0, 1.16, '$F_1$: \nMacro $F_1$: ', va='center', ha='center', 
            transform=axes[0].transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center')

    save_fig(fig, 'figs/crf.pdf')
    subprocess.call('./send_figures')


def crf_entity_result():
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

    for i, (ax, buildings) in enumerate(zip(axes, building_sets)):
        print(i)
        # Config
        ylim = (-2, 105)
        xlim = (10, 205)

        # Baseline with source
        result = baseline_results[str(buildings)]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        avg_acc = result['avg_acc']
        std_acc = result['std_acc']
        avg_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Samples'
        ys = [avg_acc, avg_mf1]
        x = sample_numbers
        xtick = sample_numbers
        xtick_labels = [str(no) for no in sample_numbers]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        linestyles = [':', ':']
        if i == 2:
            data_labels = ['Baseline Accuracy w/ Source', 
                           'Baseline Macro $F_1$ w/ Source']
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

        # Baseline without source
        result = baseline_results[str((list(buildings)[-1],))]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        avg_acc = result['avg_acc']
        std_acc = result['std_acc']
        avg_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Samples'
        ys = [avg_acc, avg_mf1]
        x = sample_numbers
        xtick = sample_numbers
        xtick_labels = [str(no) for no in sample_numbers]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        linestyles = ['-.', '-.']
        if i == 2:
            data_labels = ['Baseline Accuracy w/o Source', 
                           'Baseline Macro $F_1$ w/o Source']
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
        
        if i == 2:
            ax.legend(bbox_to_anchor=(3.2, 1.45), ncol=4, frameon=False)

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
        linestyles = ['-', '-']
        if i == 2:
            data_labels = ['Scrabble Accuracy w/o Src', 
                           'Scrabble Macro $F_1$ w/o Src']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        # Scrabble with source
        buildingfix = ''.join(list(buildings) + [buildings[-1]])
        filename = 'result/crf_entity_iter_{0}_char2tagset_iter_1.json'\
                       .format(buildingfix)
        #buildingfix = ''.join([buildings[-1]] * 2)
        #filename = 'result/crf_entity_iter_{0}_char2tagset_iter_nosource1.json'\
        #               .format(buildingfix)
        if not os.path.exists(filename):
            continue
        with open(filename, 'r') as fp:
            res = json.load(fp)
        #source_num = 0
        source_num = 200 * (len(buildings) - 1)
        srcid_lens = [len(r['learning_srcids']) - source_num for r in res]
        accuracy = [r['result']['entity']['accuracy'] * 100 for r in res]
        mf1s = [r['result']['entity']['macro_f1'] * 100 for r in res]

        x = srcid_lens
        ys = [accuracy, mf1s]
        linestyles = ['-', '-']
        if i == 2:
            data_labels = ['Scrabble Accuracy', 'Scrabble Macro $F_1$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)



        """
        # scrabble
        if ''.join(buildings) == 'ebu3bbmlap_m':
            srcids_offset = 400
        else:
            srcids_offset = 200

        try:
            with open('result/crf_entity_iter_{0}.json'.format(''.join(buildings)),
                      'r') as fp:
                result = json.load(fp)[0]
        except:
            pdb.set_trace()
            continue
        zerofile = 'result/crf_entity_iter_{0}_zero.json'.format(''.join(buildings))
        if os.path.isfile(zerofile):
            with open(zerofile, 'r') as fp:
                zero_result = json.load(fp)[0]
            x_zero = [0]
            acc_zero = [zero_result['result']['entity'][0]['accuracy'] * 100]
            mf1_zero =  [zero_result['result']['entity'][0]['macro_f1'] * 100]
        else:
            x_zero = []
            acc_zero = []
            mf1_zero = []

        fivefile = 'result/crf_entity_iter_{0}_five.json'.format(''.join(list(buildings)+[buildings[-1]]))
        if os.path.isfile(fivefile):
            with open(fivefile, 'r') as fp:
                five_result = json.load(fp)[0]
            x_five = [5]
            acc_five = [five_result['result']['entity'][0]['accuracy'] * 100]
            mf1_five =  [five_result['result']['entity'][0]['macro_f1'] * 100]
            pdb.set_trace()
        else:
            x_five = []
            acc_five = []
            mf1_five = []


        x = x_zero + x_five + [len(learning_srcids) - srcids_offset for learning_srcids in
             result['learning_srcids_history'][:-1]]
        accuracy= acc_zero + acc_five + [res['accuracy'] * 100 for res in result['result']['entity']]
        mf1s = mf1_zero + mf1_five + [res['macro_f1'] * 100 for res in result['result']['entity']]
        ys = [accuracy, mf1s]
        pdb.set_trace()
        linestyles = ['-', '-']
        if i == 2:
            data_labels = ['Scrabble Accuracy', 'Scrabble Macro $F_1$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        if i == 2:
            ax.legend(bbox_to_anchor=(3.2, 1.45), ncol=4, frameon=False)
        plot_list.append(plot)
        """


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
        
def word_sim_comp():
    buildings = ['ebu3b', 'ap_m', 'bml', 'ghc']
    word_sim_dict = dict()
    token_sim_dict = dict()
    adder = lambda x,y: x + y
    for b1 in buildings:
        for b2 in buildings:
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_words = set(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b1_s_dict.values()]))
            b2_words = set(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b2_s_dict.values()]))
            word_sim_dict['#'.join([b1, b2])] = len(b1_words.intersection(b2_words)) / \
                                          len(b2_words)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_tokens = set([s for s in reduce(adder, b1_s_dict.values()) if s.isalpha()])
            b2_tokens = set([s for s in reduce(adder, b2_s_dict.values()) if s.isalpha()])
            token_sim_dict['#'.join([b1, b2])] = len(b1_tokens.intersection(b2_tokens)) / \
                                          len(b2_tokens)
    with open('result/word_sim.json', 'w') as fp:
        json.dump(word_sim_dict, fp)
    with open('result/token_sim.json', 'w') as fp:
        json.dump(token_sim_dict, fp)

    for b1 in buildings:
        for b2 in buildings:
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_words = set(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b1_s_dict.values()]))
            b2_words = list(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b2_s_dict.values()]))
            word_sim_dict['#'.join([b1, b2])] = len([1 for w in b2_words if w in b1_words]) / \
                                          len(b2_words)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_tokens = set([s for s in reduce(adder, b1_s_dict.values()) if s.isalpha()])
            b2_tokens = list([s for s in reduce(adder, b2_s_dict.values()) if s.isalpha()])
            token_sim_dict['#'.join([b1, b2])] = len([1 for t in b2_tokens if t in b1_tokens]) / \
                                          len(b2_tokens)
    with open('result/word_sim_weighted.json', 'w') as fp:
        json.dump(word_sim_dict, fp)
    with open('result/token_sim_weighted.json', 'w') as fp:
        json.dump(token_sim_dict, fp)


def entity_iter_result():
    source_target_list = [('ebu3b', 'ap_m'),
                          ('ebu3b', 'ap_m'),
                          #, ('ghc', 'ebu3b')
                          ]
    ts_flag = False
    eda_flag = False
    fig, axes = plt.subplots(1, len(source_target_list))
#    axes = [ax]
    cs = ['firebrick', 'deepskyblue']
    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):

        filename_template = 'result/entity_iter_{0}_{1}1.json'
        prefixes = [(''.join([target]*2), 'nosource_nosa'),
                    (''.join([target]*2), 'nosource_sa'),
                    (''.join([source, target, target]), 'source_sa')]
        linestyles = [':', '-.', '-']
        for buildingfix, optfix in prefixes:
            filename = filename_template.format(buildingfix, optfix)
            with open(filename, 'r') as fp:
                data = json.load(fp)[1:]
            sa_flag = 'X' if 'nosa' in optfix else 'O'
            src_flag = '0' if 'nosource' in optfix else '200'
            source_num = int(src_flag)
            x = [len(set(datum['learning_srcids'])) - source_num for datum in data]
            accuracy = [val * 100 for val in data[-1]['accuracy_history']]
            macro_f1 = [val * 100 for val in data[-1]['macro_f1_history']]
            ys = [accuracy, macro_f1]
            xlabel = None
            ylabel = 'Score (%)'
            xtick = range(0,205, 50)
            xtick_labels = [str(n) for n in xtick]
            ytick = range(0,102,20)
            ytick_labels = [str(n) for n in ytick]
            ylim = (ytick[0]-1, ytick[-1]+2)
            if i==0:
                legends = [
                    '{0}, SA: {1}'
                    .format(src_flag, sa_flag),
                    '{0}, SA: {1}'
                    .format(src_flag, sa_flag)
                ]
            else:
                legends = None
            title = None
            plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,\
                             xtick_labels, ytick, ytick_labels, title, ax,\
                             fig, ylim, None, legends, xtickRotate=0, \
                             linestyles=[linestyles.pop()]*len(ys), cs=cs)

    for ax in axes:
        ax.grid(True)
    for ax, (source, target) in zip(axes, source_target_list):
        #ax.set_title('{0} $\Rightarrow$ {1}'.format(
        #    anon_building_dict[source], anon_building_dict[target]))
        ax.text(0.45, 0.2, '{0} $\Rightarrow$ {1}'.format(
            anon_building_dict[source], anon_building_dict[target]),
            fontsize=11,
            transform=ax.transAxes)

    for i in range(1,len(source_target_list)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')

    ax = axes[0]
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [0,1,2,3,4,5]
    new_handles = [handles[i] for i in legend_order]
    new_labels = [labels[i] for i in legend_order]
    ax.legend(new_handles, new_labels, bbox_to_anchor=(0.15,0.96), ncol=3, frameon=False)
    plt.text(0, 1.2, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center', 
            alpha=0)

    for i, ax in enumerate(axes):
        if i != 0:
            ax.set_xlabel('')

    fig.set_size_inches(4.4,1.5)
    save_fig(fig, 'figs/entity_iter.pdf')
    subprocess.call('./send_figures')
                


def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

def str2slist(s):
    s.replace(' ', '')
    return s.split(',')

def str2ilist(s):
    s.replace(' ', '')
    return [int(c) for c in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.register('type','slist', str2slist)
    parser.register('type','ilist', str2ilist)

    parser.add_argument(choices= ['crf', 'entity', 'crf_entity', 'entity_iter',
                                  'etc', 'entity_ts', 'cls', 'word_sim'],
                        dest = 'exp_type')
    args = parser.parse_args()

    if args.exp_type == 'crf':
        crf_result()
    elif args.exp_type == 'entity':
        entity_result()
    elif args.exp_type == 'crf_entity':
        crf_entity_result()
    elif args.exp_type == 'entity_iter':
        entity_iter_result()
    elif args.exp_type == 'entity_ts':
        entity_ts_result()
    elif args.exp_type == 'cls':
        cls_comp_result()
    elif args.exp_type == 'etc':
        etc_result()
    elif args.exp_type == 'word_sim':
        word_sim_comp()

