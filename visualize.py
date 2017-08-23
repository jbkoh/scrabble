import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes

import plotter

def oxer(b):
    if b:
        return 'O'
    else:
        return 'X'

def entity_ts_result():
    source_target_list = [('ebu3b', 'ap_m')]
    n_list_list = [(200,5)]
    ts_flag = False
    eda_flag = False
    inc_num = 20
    iter_num = 10
    default_query = {
        'metadata.label_type': 'label',
        'metadata.token_type': 'justseparate',
        'metadata.use_cluster_flag': True,
        'metadata.building_list' : [],
        'metadata.source_sample_num_list': [],
        'metadata.target_building': '',
        'metadata.ts_flag': ts_flag,
        'metadata.eda_flag': eda_flag,
        'metadata.use_brick_flag': True,
        'metadata.negative_flag': True,
        'metadata.inc_num': inc_num,
    }
    query_list = [deepcopy(default_query),
                  deepcopy(default_query)]
    query_list[0]['metadata.ts_flag'] = True
    fig, ax = plt.subplots(1, len(source_target_list))
    axes = [ax]
    cs = ['firebrick', 'deepskyblue']
    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = [':', '-.', '-']
        for query in query_list:
            for ns in n_list_list:
                if query['metadata.use_brick_flag'] and ns[0]==0:
                    continue
                n_s = ns[0]
                if i==1 and ns[1]==5:
                    n_t = 5
                else:
                    n_t = ns[1]

                if n_s == 0:
                    building_list = [target]
                    source_sample_num_list = [n_t]
                elif n_t == 0:
                    building_list = [source]
                    source_sample_num_list = [n_s]
                else:
                    building_list = [source, target]
                    source_sample_num_list = [n_s, n_t]
                query['metadata.building_list'] = building_list
                query['metadata.source_sample_num_list'] = \
                        source_sample_num_list
                query['metadata.target_building'] = target
                q = {'$and': [query, {'$where': \
                                      'this.accuracy_history.length=={0}'\
                                      .format(iter_num)}]}

                result = get_entity_results(q)
                try:
                    assert result
                except:
                    print(n_t)
                    pdb.set_trace()
                    result = get_entity_results(query)
                #point_precs = result['point_precision_history'][-1]
                #point_recall = result['point_recall'][-1]
                subset_accuracy_list = [val * 100 for val in result['subset_accuracy_history']]
                accuracy_list = [val * 100 for val in result['accuracy_history']]
                hierarchy_accuracy_list = [val * 100 for val in result['hierarchy_accuracy_history']]
                weighted_f1_list = [val * 100 for val in result['weighted_f1_history']]
                macro_f1_list = [val * 100 for val in result['macro_f1_history']]
                exp_num = len(macro_f1_list)
                target_n_list = list(range(n_t, inc_num*exp_num+1, inc_num))

                xs = target_n_list
                ys = [accuracy_list, macro_f1_list]
                #xlabel = '# of Target Building Samples'
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
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag'])),
                        '{0}, SA: {1}'
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag']))
                    ]
                else:
                    legends = None
                title = None
                plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
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
    #handles, labels = ax.get_legend_handles_labels()
    #legend_order = [0,1,2,3,4,5]
    #new_handles = [handles[i] for i in legend_order]
    #new_labels = [labels[i] for i in legend_order]
    #ax.legend(new_handles, new_labels, bbox_to_anchor=(0.15,0.96), ncol=3, frameon=False)
    plt.text(0, 1.2, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center', 
            alpha=0)

    for i, ax in enumerate(axes):
        if i != 0:
            ax.set_xlabel('')

    fig.set_size_inches(4.4,1.5)
    save_fig(fig, 'figs/entity_ts.pdf')
    subprocess.call('./send_figures')

def entity_iter_result():
    source_target_list = [('ebu3b', 'bml'), ('ghc', 'ebu3b')]
    n_list_list = [(200,5),
                   (0,5),]
#                   (1000,1)]
    ts_flag = False
    eda_flag = False
    inc_num = 20
    iter_num = 10
    default_query = {
        'metadata.label_type': 'label',
        'metadata.token_type': 'justseparate',
        'metadata.use_cluster_flag': True,
        'metadata.building_list' : [],
        'metadata.source_sample_num_list': [],
        'metadata.target_building': '',
        'metadata.ts_flag': ts_flag,
        'metadata.eda_flag': eda_flag,
        'metadata.use_brick_flag': True,
        'metadata.negative_flag': True,
        'metadata.inc_num': inc_num,
    }
    query_list = [deepcopy(default_query),
                  deepcopy(default_query)]
    query_list[1]['metadata.negative_flag'] = False
    query_list[1]['metadata.use_brick_flag'] = False
    fig, axes = plt.subplots(1, len(source_target_list))
#    axes = [ax]
    cs = ['firebrick', 'deepskyblue']
    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = [':', '-.', '-']
        for query in query_list:
            for ns in n_list_list:
                if query['metadata.use_brick_flag'] and ns[0]==0:
                    continue
                n_s = ns[0]
                if i==1 and ns[1]==5:
                    n_t = 5
                else:
                    n_t = ns[1]

                if n_s == 0:
                    building_list = [target]
                    source_sample_num_list = [n_t]
                elif n_t == 0:
                    building_list = [source]
                    source_sample_num_list = [n_s]
                else:
                    building_list = [source, target]
                    source_sample_num_list = [n_s, n_t]
                query['metadata.building_list'] = building_list
                query['metadata.source_sample_num_list'] = \
                        source_sample_num_list
                query['metadata.target_building'] = target
                q = {'$and': [query, {'$where': \
                                      'this.accuracy_history.length=={0}'\
                                      .format(iter_num)}]}

                result = get_entity_results(q)
                try:
                    assert result
                except:
                    print(n_t)
                    pdb.set_trace()
                    result = get_entity_results(query)
                #point_precs = result['point_precision_history'][-1]
                #point_recall = result['point_recall'][-1]
                subset_accuracy_list = [val * 100 for val in result['subset_accuracy_history']]
                accuracy_list = [val * 100 for val in result['accuracy_history']]
                hierarchy_accuracy_list = [val * 100 for val in result['hierarchy_accuracy_history']]
                weighted_f1_list = [val * 100 for val in result['weighted_f1_history']]
                macro_f1_list = [val * 100 for val in result['macro_f1_history']]
                exp_num = len(macro_f1_list)
                target_n_list = list(range(n_t, inc_num*exp_num+1, inc_num))

                xs = target_n_list
                ys = [accuracy_list, macro_f1_list]
                #xlabel = '# of Target Building Samples'
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
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag'])),
                        '{0}, SA: {1}'
                        .format(n_s,
                                oxer(query['metadata.use_brick_flag']))
                    ]
                else:
                    legends = None
                title = None
                plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                                 xtick_labels, ytick, ytick_labels, title, ax,\
                                 fig, ylim, None, legends, xtickRotate=0, \
                                 linestyles=[linestyles.pop()]*len(ys), cs=cs)
                pdb.set_trace()


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

def entity_result_deprecated():
    source_target_list = [('ebu3b', 'ap_m')]#, ('ap_m', 'ebu3b')]
    n_list_list = [[(0,5), (0,50), (0,100), (0,150), (0,200)],
                   [(200,5), (200,50), (200,100), (0,150), (200,200)]]
    ts_flag = False
    eda_flag = False
    default_query = {
        'metadata.label_type': 'label',
        'metadata.token_type': 'justseparate',
        'metadata.use_cluster_flag': True,
        'metadata.building_list' : [],
        'metadata.source_sample_num_list': [],
        'metadata.target_building': '',
        'metadata.ts_flag': ts_flag,
        'metadata.eda_flag': eda_flag,
        'metadata.use_brick_flag': True
    }
    query_list = [deepcopy(default_query),\
                 deepcopy(default_query),\
                 deepcopy(default_query)]
    query_list[0]['metadata.use_brick_flag'] = False
    query_list[0]['metadata.negative_flag'] = False
    query_list[1]['metadata.use_brick_flag'] = False
    query_list[1]['metadata.negative_flag'] = True
    query_list[2]['metadata.use_brick_flag'] = True 
    query_list[2]['metadata.negative_flag'] = True
    char_precs_list = list()
    phrase_f1s_list = list()
    fig, axes = plt.subplots(1, 3)
#axes = [ax]
    fig.set_size_inches(8,5)
    #fig, axes = plt.subplots(1,len(n_list_list))

    for ax, (source, target) in zip(axes, source_target_list):
        for query in query_list:
            for n_list in n_list_list:
                target_n_list = [ns[1] for ns in n_list]
                subset_accuracy_list = list()
                accuracy_list = list()
                hierarchy_accuracy_list = list()
                weighted_f1_list = list()
                macro_f1_list = list()

                for (n_s, n_t) in n_list:
                    if n_s == 0:
                        building_list = [target]
                        source_sample_num_list = [n_t]
                    elif n_t == 0:
                        building_list = [source]
                        source_sample_num_list = [n_s]
                    else:
                        building_list = [source, target]
                        source_sample_num_list = [n_s, n_t]
                    query['metadata.building_list'] = building_list
                    query['metadata.source_sample_num_list'] = \
                            source_sample_num_list
                    query['metadata.target_building'] = target

                    result = get_entity_results(query)
                    try:
                        assert result
                    except:
                        print(n_t)
                        pdb.set_trace()
                        result = get_entity_results(query)
                    #point_precs = result['point_precision_history'][-1]
                    #point_recall = result['point_recall'][-1]
                    subset_accuracy_list.append(result['subset_accuracy_history'][-1] * 100)
                    accuracy_list.append(result['accuracy_history'][-1] * 100)
                    hierarchy_accuracy_list.append(result['hierarchy_accuracy_history'][-1] * 100)
                    weighted_f1_list.append(result['weighted_f1_history'][-1] * 100)
                    macro_f1_list.append(result['macro_f1_history'][-1] * 100)

                xs = target_n_list
                ys = [hierarchy_accuracy_list, accuracy_list, macro_f1_list]
                xlabel = '# of Target Building Samples'
                ylabel = 'Score (%)'
                xtick = target_n_list
                xtick_labels = [str(n) for n in target_n_list]
                ytick = range(0,102,10)
                ytick_labels = [str(n) for n in ytick]
                ylim = (ytick[0]-1, ytick[-1]+2)
                legends = [
                    '{0}, SA:{1}'\
                    .format(n_s, query['metadata.use_brick_flag']),
                    '{0}, SA:{1}'\
                    .format(n_s, query['metadata.use_brick_flag']),
                    '{0}, SA:{1}'\
                    .format(n_s, query['metadata.use_brick_flag'])
                          ]
                title = None
                plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                                 xtick_labels, ytick, ytick_labels, title, ax, fig, \
                                 ylim, legends)
                #plotter.plot_multiple_2dline(xs, [ys[1]], xlabel, ylabel, xtick,\
                #                 xtick_labels, ytick, ytick_labels, title, axes[1], fig, \
                #                 ylim, [legends[1]])
                #plotter.plot_multiple_2dline(xs, [ys[2]], xlabel, ylabel, xtick,\
                #                 xtick_labels, ytick, ytick_labels, title, axes[2], fig, \
                #                 ylim, [legends[2]])
                if not (query['metadata.negative_flag'] and
                        query['metadata.use_brick_flag']):
                    break
    axes[0].set_title('Hierarchical Accuracy')
    axes[1].set_title('Accuracy')
    axes[2].set_title('Macro F1')
    suptitle = 'Multi Label (TagSets) Classification with a Source building.'
    fig.suptitle(suptitle)
    save_fig(fig, 'figs/entity.pdf')

def crf_entity_result():
    building_sets = [('ebu3b', 'ap_m'), ('ap_m', 'bml'),
                 ('ebu3b', 'ghc'), ('ghc', 'ebu3b'), ('ebu3b', 'bml', 'ap_m')] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    fig, axes = plt.subplots(1, len(building_sets))
    with open('result/baseline.json', 'r') as fp:
        baseline_results = json.load(fp)

    cs = ['firebrick', 'deepskyblue']
    plot_list = list()

    for i, (ax, buildings) in enumerate(zip(axes, building_sets)):
        print(i)
        # Baseline
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
        ylim = (-2, 105)
        xlim = (10, 205)
        linestyles = [':', ':']
        if i == 2:
            data_labels = ['Baseline Accuracy', 'Baseline Macro $F_1$']
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
        pdb.set_trace()


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

def cls_comp_result():
    source_target_list = ('ebu3b', 'ap_m')
    keys = ['best', 'ts', 'rf', 'svc']
    xs = list(range(5, 205, 20))
    accuracy_dict = OrderedDict({
        'best':[0.8631033290671848, 0.9024136840401907, 0.9233413507509902, 0.9500121364579196, 0.9527101078305895, 0.9650918693087369, 0.9677129764479163, 0.9593822175147483, 0.9711269988378419, 0.9697809553231241],
        'ts': [0.8713471602025806, 0.9166264458433141, 0.9185595580405604, 0.9428053539499326, 0.9417577736854855, 0.9573296850405294, 0.9489047766156204, 0.9534092413610498, 0.953262734588037, 0.9595308306151684],
        'rf': [0.756387822351681, 0.854248495814764, 0.8465179398914331, 0.859092781381938, 0.9137193462494689, 0.9384494020036196, 0.9460637421480792, 0.9512496873942656, 0.9582711799579264, 0.9597065919355077],
        'svc': [0.7210336660658784, 0.8278964869103078, 0.8371634459716821, 0.8901948134091584, 0.9289625735354351, 0.9062837090984304, 0.9072457164626379, 0.9094597402145658, 0.9061470144946531, 0.9317219571018263]
    })
    mf1_dict = OrderedDict({
        'best': [0.43460544517064525, 0.46207967166726716, 0.60572075680286364, 0.65253670730553948, 0.71164857967833528, 0.77075401369085861, 0.77409145497551546, 0.78223293415400674, 0.79434165930991263, 0.78765666427863568],
        'ts': [0.38456663841099153, 0.47135950957306999, 0.50801383768831809, 0.58379558680943822, 0.61765049559624907, 0.67617354377548211, 0.66706236361751792, 0.70816840695824457, 0.68736126966336153, 0.70501274992734486],
        'rf': [0.094018355593671443, 0.21622362914898177, 0.2939715246436253, 0.38083088857608816, 0.45237091518218492, 0.51912845475805691, 0.56752106411334313, 0.6314794515347395, 0.73066778675441313, 0.81505177770253923],
        'svc': [0.19122967879394315, 0.2501766458806039, 0.27629897715774632, 0.31374977389303144, 0.35811497520318963, 0.36814352938387473, 0.37145631338451729, 0.38910680891542943, 0.35511959588962361, 0.39688667191674587]
    })
    legends = ['SCRBL', 'w/ TS', 'RF', 'w/ SVC'] * 2
    linestyles = ['-', ':', '-.', '--'] * 2
    cs = ['firebrick']*len(keys) + ['deepskyblue'] * len(keys)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(4,1.7)
    axes = [ax]
    mult = lambda x: x*100
    hundreder = lambda seq: list(map(mult, seq))
    ys = list(map(hundreder, list(accuracy_dict.values()) + list(mf1_dict.values())))
    #ys = [char_precs, phrase_f1s, char_macro_f1s, phrase_macro_f1s]
    xlabel = '# of Target Building Samples'
    ylabel = 'Score (%)'
    xtick = list(range(0, 205, 20))
    xtick_labels = [str(n) for n in xtick]
    ytick = range(0,101,20)
    ytick_labels = [str(n) for n in ytick]
    xlim = (xtick[0]-2, xtick[-1]+5)
    ylim = (ytick[0]-2, ytick[-1]+5)
    title = None
    _, plots = plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                            xtick_labels, ytick, ytick_labels, title, ax, fig, \
                            ylim, xlim, None , xtickRotate=0, \
                            linestyles=linestyles, cs=cs)
   #ax.legend(plots, legends, 'upper center', ncol=4
    legend_order = [0,4,1,5,2,6,3,7]
    new_handles = [plots[i] for i in legend_order]
    new_legends = [legends[i] for i in legend_order]
    fig.legend(new_handles, new_legends, ncol=4, bbox_to_anchor=(-0.1, 1.04, 1, 0.095),
               prop={'size':7}, frameon=False )
    for ax in axes:
        ax.grid(True)
    plt.text(0.03, 1.135, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes, fontsize=7)
    save_fig(fig, 'figs/cls.pdf')
    subprocess.call('./send_figures')



def crf_result():
    source_target_list = [('ebu3b', 'bml'), ('ghc', 'ebu3b')]
    n_list_list = [[(1000, 0), (1000,5), (1000,20), (1000,50), (1000,100), (1000, 150), (1000,200)],
                   [(200, 0), (200,5), (200,20), (200,50), (200,100), (200, 150), (200,200)],
                   [(0,5), (0,20), (0,50), (0,100), (0,150), (0,200)]]
    char_precs_list = list()
    phrase_f1s_list = list()
#fig, ax = plt.subplots(1, 1)
    fig, axes = plt.subplots(1,len(source_target_list))
    if isinstance(axes, Axes):
        axes = [axes]
    fig.set_size_inches(4, 1.5)
    cs = ['firebrick', 'deepskyblue']

    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = ['--', '-.', '-']
        plot_list = list()
        legends_list = list()
        for n_list in n_list_list:
            target_n_list = [ns[1] for ns in n_list]
            phrase_f1s = list()
            char_macro_f1s = list()
            phrase_macro_f1s = list()
#pess_phrase_f1s = list()
            char_precs = list()
            for (n_s, n_t) in n_list:
                if n_s == 0:
                    building_list = [target]
                    source_sample_num_list = [n_t]
                elif n_t == 0:
                    building_list = [source]
                    source_sample_num_list = [n_s]
                else:
                    building_list = [source, target]
                    source_sample_num_list = [n_s, n_t]
                result_query = {
                    'label_type': 'label',
                    'token_type': 'justseparate',
                    'use_cluster_flag': True,
                    'building_list': building_list,
                    'source_sample_num_list': source_sample_num_list,

                    'target_building': target
                }
                result = get_crf_results(result_query)
                try:
                    assert result
                except:
                    print(n_t)
                    pdb.set_trace()
                    continue
                    result = get_crf_results(result_query)
                char_prec = result['char_precision'] * 100
                char_precs.append(char_prec)
                phrase_recall = result['phrase_recall'] * 100
                phrase_prec = result['phrase_precision'] * 100
                phrase_f1 = 2* phrase_prec  * phrase_recall \
                                / (phrase_prec + phrase_recall)
                phrase_f1s.append(phrase_f1)
                char_macro_f1s.append(result['char_macro_f1'] * 100)
                phrase_macro_f1s.append(result['phrase_macro_f1'] * 100)
            xs = target_n_list
            ys = [phrase_f1s, phrase_macro_f1s]
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
            if i == 0:
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
            pdb.set_trace()

#fig.legend(plot_list, legends_list, 'upper center', ncol=3
#            , bbox_to_anchor=(0.5,1.3),frameon=False)
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

def etc_result():
    buildings = ['ebu3b', 'bml', 'ap_m', 'ghc']
    tagsets_dict = dict()
    tags_dict = dict()
    tagset_numbers = []
    avg_required_tags = []
    tagset_type_numbers = []
    tags_numbers = []
    tags_type_numbers = []
    avg_tagsets = []
    median_occ_numbers = []
    avg_tags = []
    avg_tokens = []#TODO: Need to use words other than tokens
    avg_unfound_tags = []#TODO: Need to use words other than tokens
    once_numbers = []
    top20_numbers = []
    std_tags = []
    std_tagsets = []

    ignore_tagsets = ['leftidentifier', 'rightidentifier', 'none', 'unknown']
    total_tagsets = list(set(point_tagsets + location_tagsets + equip_tagsets))
    total_tags = list(set(reduce(adder, map(splitter, total_tagsets))))

    for building in buildings:
        with open('metadata/{0}_ground_truth.json'\
                  .format(building), 'r') as fp:
            truth_dict = json.load(fp)
        with open('metadata/{0}_label_dict_justseparate.json'.\
                  format(building), 'r') as fp:
            label_dict = json.load(fp)
        with open('metadata/{0}_sentence_dict_justseparate.json'\
                  .format(building), 'r') as fp:
            sentence_dict = json.load(fp)
        new_label_dict = dict()
        for srcid, labels in label_dict.items():
            new_label_dict[srcid] = list(reduce(adder, [label.split('_')
                                                   for label in labels if label
                                                   not in ignore_tagsets]))
        label_dict = new_label_dict
        srcids = list(label_dict.keys())
        label_dict = OrderedDict([(srcid, label_dict[srcid])
                                 for srcid in srcids])
        truth_dict = OrderedDict([(srcid, truth_dict[srcid])
                                 for srcid in srcids])
        sentence_dict = OrderedDict([(srcid, sentence_dict[srcid])
                                    for srcid in srcids])

        tagsets = [tagset for tagset in
                   list(reduce(adder, truth_dict.values()))
                   if tagset not in ignore_tagsets]
        def tagerize(tagsets):
            return list(set(reduce(adder, map(splitter, tagsets))))
        required_tags = list(map(tagerize, truth_dict.values()))
        tags = list(reduce(adder, map(splitter, tagsets)))
        tagsets_counter = Counter(tagsets)
        tagsets_dict[building] = Counter(tagsets)
        tagset_numbers.append(len(tagsets))
        tags_numbers.append(len(tags))
        tagset_type_numbers.append(len(set(tagsets)))
        tags_type_numbers.append(len(set(tags)))
        tokens_list = [[token for token in tokens
                        if re.match('[a-zA-Z]+', token)]
                       for tokens in sentence_dict.values()]
        unfound_tags_list = list()
        for srcid, tagsets in truth_dict.items():
            unfound_tags = set()
            for tagset in tagsets:
                for tag in tagset.split('_'):
                    if tag not in label_dict[srcid]:
                        unfound_tags.add(tag)
            unfound_tags_list.append(unfound_tags)
        avg_tokens.append(np.mean(list(map(lengther, tokens_list))))
        avg_tags.append(np.mean(list(map(lengther, map(set,label_dict.values())))))
        std_tags.append(np.std(list(map(lengther, map(set,label_dict.values())))))
        avg_tagsets.append(np.mean(list(map(lengther, truth_dict.values()))))
        std_tagsets.append(np.std(list(map(lengther, truth_dict.values()))))
        avg_required_tags.append(np.mean(list(map(lengther, required_tags))))
        avg_unfound_tags.append(np.mean(list(map(lengther, unfound_tags_list))))
        once_occurring_tagsets = [tagset for tagset, cnt
                                 in tagsets_counter.items() if cnt==1]

        once_numbers.append(len(once_occurring_tagsets))
        top20_numbers.append(np.sum(sorted(tagsets_counter.values(),
                                           reverse=True)[0:20]))
        median_occ_numbers.append(np.median(list(tagsets_counter.values())))

    tags_cnt = 0
    for tagset in total_tagsets:
        tags_cnt += len(splitter(tagset))
    avg_len_tagset = tags_cnt / len(total_tagsets)
    print('tot tags :', tagset_numbers)
    print('tot tagsets:', tags_numbers)
    print('avg len tagset:', avg_len_tagset)
    print('avg tokens:', avg_tokens)
    print('avg tags :', avg_tags)
    print('std tags :', std_tags)
    print('avg tagsets:', avg_tagsets)
    print('std tagsets:', std_tagsets)
    print('avg required tags:', avg_required_tags)
    print('avg unfound tags:', avg_unfound_tags)
    print('tot tagset:', tagset_type_numbers)
    print('tot tags:', tags_type_numbers)
    print('top20 explains: ', top20_numbers)
    print('num once occur: ', once_numbers)
    print('median occs: ', median_occ_numbers)
    print('brick tagsets: ', len(total_tagsets))
    print('brick tags: ', len(total_tags))
