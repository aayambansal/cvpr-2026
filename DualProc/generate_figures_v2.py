#!/usr/bin/env python3
"""Generate all v2 figures for the expanded DualProc paper."""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'axes.linewidth': 0.8,
})

# Okabe-Ito palette
C = {'baseline': '#E69F00', 'cot': '#56B4E9', 'dual_process': '#009E73',
     'deliberate_only': '#CC79A7', 'self_consistency': '#F0E442',
     'self_refine': '#D55E00', 'verbal_calibration': '#0072B2',
     'temp_scaling': '#999999'}

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'experiment_results_v2.json')) as f:
    D = json.load(f)
fig_dir = os.path.join(script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

MODELS_MAIN = ['GPT-4o-mini', 'Gemini-2.0-Flash', 'Claude-3.5-Sonnet']
MODELS_ALL = D['metadata']['models']
CONDITIONS = D['metadata']['conditions']
COND_LABELS = D['metadata']['condition_labels']
SHORT = {'GPT-4o-mini': 'GPT-4o-mini', 'Gemini-2.0-Flash': 'Gemini Flash',
         'Claude-3.5-Sonnet': 'Claude 3.5', 'LLaVA-1.6-34B': 'LLaVA-1.6',
         'InternVL2-26B': 'InternVL2'}
CATS = D['metadata']['category_labels']


def save(fig, name):
    fig.savefig(os.path.join(fig_dir, name))
    plt.close(fig)
    print(f"  + {name}")


# ======================================================================
# Fig 1: Method overview (same as before but refined)
# ======================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis('off')
    boxes = [(0.3,1.2,2.4,1.6,'#FFF3CD','Stage 1: System 1\n(Fast Guess)','Quick answer\n+ confidence $c_1$'),
             (3.5,1.2,2.4,1.6,'#D4EDDA','Stage 2: Deliberation','3 alternatives\n+ evidence check'),
             (6.8,1.2,2.4,1.6,'#CCE5FF','Stage 3: System 2\n(Final Answer)','Revised answer\n+ confidence $c_2$')]
    for x,y,w,h,c,t,s in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",facecolor=c,edgecolor='#333',lw=1.2))
        ax.text(x+w/2,y+h*0.68,t,ha='center',va='center',fontsize=9,fontweight='bold',color='#222')
        ax.text(x+w/2,y+h*0.28,s,ha='center',va='center',fontsize=7.5,color='#555',style='italic')
    for xy1,xy2 in [((2.8,2.0),(3.4,2.0)),((6.0,2.0),(6.7,2.0))]:
        ax.annotate('',xy=xy2,xytext=xy1,arrowprops=dict(arrowstyle='->',color='#555',lw=1.5))
    ax.text(5.0,3.2,'DualProc: Deliberation selectively targets confident errors, '
            'improving calibration without sacrificing accuracy',
            ha='center',va='center',fontsize=8,style='italic',
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='#009E73',lw=1))
    ax.text(1.5,0.85,'$c_1 = 0.85$',ha='center',fontsize=8,color='#B8860B')
    ax.text(8.0,0.85,'$c_2 = 0.55$',ha='center',fontsize=8,color='#2171B5')
    save(fig, 'fig1_method_overview.pdf')


# ======================================================================
# Fig 2: Main results – 8 methods, 5 models (compact grouped bar)
# ======================================================================
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
    metrics = [('accuracy','Accuracy $\\uparrow$',(0.5,0.85)),
               ('confidence_gap','Conf. Gap $\\downarrow$',(-0.06,0.20)),
               ('confident_error_rate','Conf. Error Rate $\\downarrow$',(0,0.22))]
    x = np.arange(len(MODELS_ALL))
    w = 0.09
    for ai,(mk,yl,ylim) in enumerate(metrics):
        ax = axes[ai]
        for ci, cond in enumerate(CONDITIONS):
            vals = [D['core_results'][m][cond][mk] for m in MODELS_ALL]
            ax.bar(x+(ci-3.5)*w, vals, w, color=C[cond], edgecolor='white', lw=0.3)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS_ALL],fontsize=6,rotation=25,ha='right')
        ax.set_ylabel(yl,fontsize=8); ax.set_ylim(ylim)
        ax.grid(axis='y',alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if mk=='confidence_gap': ax.axhline(0,color='k',lw=0.5)
    handles = [mpatches.Patch(color=C[c],label=COND_LABELS[c]) for c in CONDITIONS]
    axes[0].legend(handles=handles,fontsize=5,ncol=2,loc='lower left')
    plt.tight_layout()
    save(fig, 'fig2_main_results.pdf')


# ======================================================================
# Fig 3: Reliability diagrams
# ======================================================================
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for mi,m in enumerate(MODELS_MAIN):
        ax = axes[mi]
        for cond in ['baseline','cot','dual_process','temp_scaling']:
            bins = D['core_results'][m][cond]['ece_bins']
            cs = [b['avg_conf'] for b in bins if b['count']>0]
            ac = [b['avg_acc'] for b in bins if b['count']>0]
            ax.plot(cs,ac,'o-',color=C[cond],label=COND_LABELS[cond],markersize=3,lw=1.2)
        ax.plot([0,1],[0,1],'k--',lw=0.8,alpha=0.5)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
        ax.set_xlabel('Confidence',fontsize=8); ax.set_title(SHORT[m],fontsize=9,fontweight='bold')
        ax.grid(alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if mi==0: ax.set_ylabel('Accuracy',fontsize=8); ax.legend(fontsize=5.5,loc='upper left')
    plt.tight_layout()
    save(fig, 'fig3_calibration.pdf')


# ======================================================================
# Fig 4: Agentic loop results
# ======================================================================
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    agent_metrics = [
        ('task_completion_rate', 'Task Completion', (0.6, 1.0)),
        ('tool_misuse_rate', 'Tool Misuse Rate $\\downarrow$', (0, 0.20)),
        ('safe_deferral_rate', 'Safe Deferral Rate', (0, 0.15)),
    ]
    agent_types = ['baseline_agent', 'cot_agent', 'dualproc_agent']
    agent_colors = ['#E69F00', '#56B4E9', '#009E73']
    agent_labels = ['Baseline Agent', 'CoT Agent', 'DualProc Agent']
    x = np.arange(len(MODELS_ALL))
    w = 0.22

    for ai, (mk, yl, ylim) in enumerate(agent_metrics):
        ax = axes[ai]
        for ti, at in enumerate(agent_types):
            vals = [D['agentic_results'][m][at][mk] for m in MODELS_ALL]
            ax.bar(x + (ti-1)*w, vals, w, color=agent_colors[ti], edgecolor='white', lw=0.3)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS_ALL], fontsize=6, rotation=25, ha='right')
        ax.set_ylabel(yl, fontsize=8); ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(agent_colors, agent_labels)]
    axes[0].legend(handles=handles, fontsize=6, loc='lower left')
    plt.tight_layout()
    save(fig, 'fig4_agentic_loop.pdf')


# ======================================================================
# Fig 5: Grounded retrieval results
# ======================================================================
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    pipelines = ['baseline_no_retrieval', 'retrieval_only', 'retrieval_plus_dualproc']
    pipe_labels = ['No Retrieval', 'Retrieval Only', 'Retrieval+DualProc']
    pipe_colors = ['#E69F00', '#56B4E9', '#009E73']
    ret_metrics = [
        ('accuracy', 'Accuracy', (0.55, 0.80)),
        ('confident_error_rate', 'Conf. Error Rate $\\downarrow$', (0, 0.16)),
        ('mean_evidence_precision', 'Evidence Precision', (0, 0.65)),
    ]
    x = np.arange(len(MODELS_ALL))
    w = 0.22
    for ai, (mk, yl, ylim) in enumerate(ret_metrics):
        ax = axes[ai]
        for pi, p in enumerate(pipelines):
            vals = [D['retrieval_results'][m][p][mk] for m in MODELS_ALL]
            ax.bar(x + (pi-1)*w, vals, w, color=pipe_colors[pi], edgecolor='white', lw=0.3)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS_ALL], fontsize=6, rotation=25, ha='right')
        ax.set_ylabel(yl, fontsize=8); ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(pipe_colors, pipe_labels)]
    axes[0].legend(handles=handles, fontsize=5.5, loc='lower left')
    plt.tight_layout()
    save(fig, 'fig5_grounded_retrieval.pdf')


# ======================================================================
# Fig 6: Adaptive prompting Pareto frontier
# ======================================================================
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    markers = {'GPT-4o-mini': 'o', 'Gemini-2.0-Flash': 's', 'Claude-3.5-Sonnet': '^'}

    for mi, m in enumerate(MODELS_MAIN):
        thetas = sorted(D['adaptive_results'][m].keys())
        tokens = [D['adaptive_results'][m][t]['avg_tokens_per_item'] for t in thetas]
        cers = [D['adaptive_results'][m][t]['confident_error_rate'] for t in thetas]
        accs = [D['adaptive_results'][m][t]['accuracy'] for t in thetas]

        axes[0].plot(tokens, cers, 'o-', color=C['dual_process'] if mi==0 else C['cot'] if mi==1 else C['baseline'],
                    marker=markers[m], markersize=5, lw=1.2, label=SHORT[m])
        axes[1].plot(tokens, accs, 'o-', color=C['dual_process'] if mi==0 else C['cot'] if mi==1 else C['baseline'],
                    marker=markers[m], markersize=5, lw=1.2, label=SHORT[m])

    for ax_i, (ax, yl) in enumerate([(axes[0], 'Conf. Error Rate $\\downarrow$'), (axes[1], 'Accuracy $\\uparrow$')]):
        ax.set_xlabel('Tokens per Item', fontsize=9); ax.set_ylabel(yl, fontsize=9)
        ax.grid(alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7)

    # Annotate sweet spot
    axes[0].annotate('Sweet spot\n($\\theta=0.60$)', xy=(918, 0.008), xytext=(700, 0.06),
                    fontsize=7, arrowprops=dict(arrowstyle='->', color='#009E73'), color='#009E73')
    plt.tight_layout()
    save(fig, 'fig6_adaptive_pareto.pdf')


# ======================================================================
# Fig 7: Checklist ablation
# ======================================================================
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ablation_names = ['full_checklist', '1_alternative', '5_alternatives',
                      'no_evidence_check', 'no_missed_details', 'alternatives_only', 'evidence_only']
    short_names = ['Full', '1-alt', '5-alt', 'No-evid', 'No-miss', 'Alt-only', 'Evid-only']
    colors_abl = ['#009E73', '#56B4E9', '#56B4E9', '#D55E00', '#D55E00', '#E69F00', '#E69F00']

    # Average across 3 models
    avg_acc = [np.mean([D['ablation_results'][m][a]['accuracy'] for m in MODELS_MAIN]) for a in ablation_names]
    avg_cer = [np.mean([D['ablation_results'][m][a]['confident_error_rate'] for m in MODELS_MAIN]) for a in ablation_names]
    avg_sep = [np.mean([D['ablation_results'][m][a]['conf_separation'] for m in MODELS_MAIN]) for a in ablation_names]

    x = np.arange(len(ablation_names))
    axes[0].bar(x, avg_acc, color=colors_abl, edgecolor='white', lw=0.5)
    axes[0].set_ylabel('Accuracy (avg 3 models)', fontsize=8)
    axes[0].set_ylim(0.65, 0.78)

    axes[1].bar(x, avg_cer, color=colors_abl, edgecolor='white', lw=0.5)
    axes[1].set_ylabel('Conf. Error Rate (avg)', fontsize=8)

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, 'fig7_checklist_ablation.pdf')


# ======================================================================
# Fig 8: Multi-seed robustness (box plots)
# ======================================================================
def fig8():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for mi, m in enumerate(MODELS_MAIN):
        ax = axes[mi]
        rob = D['robustness_results'][m]
        positions = [0, 1, 2]
        conds_rob = ['baseline', 'cot', 'dual_process']
        box_data = []
        for cond in conds_rob:
            accs = [s['accuracy'] for s in rob[cond]['per_seed']]
            box_data.append(accs)
        bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                       showfliers=True, flierprops=dict(markersize=3))
        for patch, cond in zip(bp['boxes'], conds_rob):
            patch.set_facecolor(C[cond]); patch.set_alpha(0.7)
        ax.set_xticks(positions); ax.set_xticklabels(['Direct', 'CoT', 'DualProc'], fontsize=7)
        ax.set_ylabel('Accuracy (10 seeds)', fontsize=8)
        ax.set_title(SHORT[m], fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # Significance annotation
        sig = rob['significance']
        if sig['accuracy_significant']:
            y_max = max(max(d) for d in box_data) + 0.01
            ax.plot([0, 0, 2, 2], [y_max, y_max+0.005, y_max+0.005, y_max], 'k-', lw=0.8)
            ax.text(1, y_max+0.008, f'p={sig["accuracy_p"]:.4f}*', ha='center', fontsize=6)
    plt.tight_layout()
    save(fig, 'fig8_robustness.pdf')


# ======================================================================
# Fig 9: Confidence separation improvement
# ======================================================================
def fig9():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x = np.arange(len(MODELS_ALL))
    w = 0.3
    base_sep = [D['core_results'][m]['baseline']['conf_separation'] for m in MODELS_ALL]
    dual_sep = [D['core_results'][m]['dual_process']['conf_separation'] for m in MODELS_ALL]
    ax.bar(x - w/2, base_sep, w, color=C['baseline'], label='Direct', edgecolor='white')
    ax.bar(x + w/2, dual_sep, w, color=C['dual_process'], label='DualProc', edgecolor='white')

    for i in range(len(MODELS_ALL)):
        pct = (dual_sep[i] - base_sep[i]) / base_sep[i] * 100
        ax.text(i, max(dual_sep[i], base_sep[i]) + 0.01, f'+{pct:.0f}%',
               ha='center', fontsize=6.5, color='#009E73', fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS_ALL], fontsize=7, rotation=20, ha='right')
    ax.set_ylabel('Conf.-Correctness Separation', fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, 'fig9_separation.pdf')


# ======================================================================
# Fig 10: Per-category accuracy heatmap (5 models)
# ======================================================================
def fig10():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    conds_show = ['baseline', 'cot', 'dual_process', 'self_refine', 'temp_scaling']
    cond_short = ['Direct', 'CoT', 'DualProc', 'Self-Ref.', 'Temp.S.']
    for mi, m in enumerate(MODELS_MAIN):
        ax = axes[mi]
        mat = np.zeros((len(conds_show), len(CATS)))
        for ci, cond in enumerate(conds_show):
            for ki, cat in enumerate(D['metadata']['categories']):
                mat[ci, ki] = D['core_results'][m][cond]['per_category'][cat]['accuracy']
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0.35, vmax=0.90, aspect='auto')
        for ci in range(len(conds_show)):
            for ki in range(len(CATS)):
                c = 'white' if mat[ci,ki] < 0.5 or mat[ci,ki] > 0.85 else 'black'
                ax.text(ki, ci, f'{mat[ci,ki]:.2f}', ha='center', va='center', fontsize=5.5, color=c)
        ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, fontsize=6, rotation=30, ha='right')
        ax.set_yticks(range(len(conds_show)))
        ax.set_yticklabels(cond_short if mi==0 else [], fontsize=6.5)
        ax.set_title(SHORT[m], fontsize=9, fontweight='bold')
    plt.tight_layout()
    save(fig, 'fig10_category_heatmap.pdf')


# ======================================================================
# Fig 11: Flip analysis (expanded)
# ======================================================================
def fig11():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x = np.arange(len(MODELS_ALL))
    w = 0.3
    fc = [D['core_results'][m]['flips_baseline_to_dual_process']['flip_to_correct_pct'] for m in MODELS_ALL]
    fw = [D['core_results'][m]['flips_baseline_to_dual_process']['flip_to_wrong_pct'] for m in MODELS_ALL]
    ax.bar(x-w/2, fc, w, color='#2CA02C', label='Flip to Correct', edgecolor='white')
    ax.bar(x+w/2, fw, w, color='#D62728', label='Flip to Wrong', edgecolor='white')
    # Add net annotation
    for i, m in enumerate(MODELS_ALL):
        fl = D['core_results'][m]['flips_baseline_to_dual_process']
        ax.text(i, max(fc[i], fw[i])+1.5, f'{fl["flip_ratio"]:.1f}x', ha='center', fontsize=6, fontweight='bold', color='#009E73')
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS_ALL], fontsize=6, rotation=20, ha='right')
    ax.set_ylabel('Items Flipped (%)', fontsize=8); ax.legend(fontsize=6.5)
    ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, 'fig11_flip_analysis.pdf')


# ======================================================================
# Fig 12: Open-source vs closed-source comparison
# ======================================================================
def fig12():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    commercial = ['GPT-4o-mini', 'Gemini-2.0-Flash', 'Claude-3.5-Sonnet']
    opensource = ['LLaVA-1.6-34B', 'InternVL2-26B']

    for ax_i, (group, title) in enumerate([(commercial, 'Commercial VLMs'), (opensource, 'Open-Source VLMs')]):
        ax = axes[ax_i]
        x = np.arange(len(group))
        w = 0.18
        for ci, cond in enumerate(['baseline', 'cot', 'dual_process', 'self_refine']):
            vals = [D['core_results'][m][cond]['confident_error_rate'] for m in group]
            ax.bar(x + (ci-1.5)*w, vals, w, color=C[cond], edgecolor='white', lw=0.3,
                  label=COND_LABELS[cond] if ax_i==0 else '')
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in group], fontsize=7)
        ax.set_ylabel('Conf. Error Rate $\\downarrow$', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].legend(fontsize=6, ncol=2, loc='upper left')
    plt.tight_layout()
    save(fig, 'fig12_open_vs_closed.pdf')


# ======================================================================
# Run all
# ======================================================================
if __name__ == '__main__':
    print("Generating v2 figures...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9(); fig10(); fig11(); fig12()
    print(f"All {12} figures saved to {fig_dir}/")
