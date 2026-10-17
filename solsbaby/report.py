"""Static scientific reports; usable without Tk or a graphical display."""
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

BACKGROUND = '#0d1420'
PANEL = '#131f2e'
TEXT = '#e5edf5'
MUTED = '#9aacc0'
COLORS = ['#65d0d6', '#efb96e', '#b59beb', '#ef91a4', '#8ccc9a', '#8ab5e5', '#df9e6a', '#c9d77d']
LABELS = {'learned_soma': 'Learned\nsoma', 'frozen_writer': 'Frozen\nwriter',
          'erased_memory': 'Erased\nmemory', 'shuffled_labels': 'Shuffled\nlabels',
          'slow_state_oracle': 'Slow-state\noracle'}


def style_axes(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#314157')
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(alpha=.12, color=MUTED)


def new_figure(title):
    fig = Figure(figsize=(13.6, 8.4), facecolor=BACKGROUND, layout='constrained')
    FigureCanvasAgg(fig)
    fig.suptitle(title, color=TEXT, fontsize=18, fontweight='bold')
    return fig


def draw_tree(ax, morphology, values, kind='tag'):
    ax.clear()
    style_axes(ax)
    m = morphology
    values = np.asarray(values)
    for i in range(1, m.n):
        points = m.xy[[m.parent[i], i]]
        ax.plot(points[:, 0], points[:, 1], color='#647890', linewidth=1+3*m.diameter[i], alpha=.65)
    image = ax.scatter(m.xy[:, 0], m.xy[:, 1], c=values, cmap='viridis',
                       norm=Normalize(0, 1), s=65, zorder=3, edgecolors=BACKGROUND, linewidth=.6)
    ax.scatter([0], [0], s=220, facecolor=COLORS[0], edgecolor=TEXT, linewidth=1.3, zorder=4)
    ax.annotate('soma', (0, 0), xytext=(7, 7), textcoords='offset points', color=TEXT, fontsize=10)
    for b, tip in enumerate(m.tips):
        ax.annotate(chr(65+b), m.xy[tip], xytext=(6, 6), textcoords='offset points',
                    color=COLORS[b], fontweight='bold', fontsize=12)
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(.18)
    ax.axis('off')
    ax.set_title('Learned shunt p' if kind == 'tag' else 'Voltage v', loc='left', pad=12, color=TEXT)
    return image


def draw_probe(ax, learned, erased, protocol):
    style_axes(ax)
    t = learned['time']
    dt = t[0]
    ax.plot(t, erased['trace'][0], color=MUTED, linestyle='--', linewidth=1.8, label='Memory erased')
    ax.plot(t, learned['trace'][0], color=COLORS[0], linewidth=2, label='After experience')
    for b in range(protocol.branches):
        left = b*protocol.slot_steps*dt
        right = (b+1)*protocol.slot_steps*dt
        ax.axvspan(left, right, color=COLORS[b], alpha=.045)
        ax.text((left+right)/2, .94, chr(65+b), transform=ax.get_xaxis_transform(),
                color=COLORS[b], ha='center', fontsize=10)
    ax.set(xlabel='Time (dimensionless)', ylabel='Soma voltage change', title='Same probe · one somatic port')
    ax.legend(facecolor=PANEL, edgecolor='#314157', labelcolor=TEXT, fontsize=9, loc='lower right')


def demo_figure(data):
    medium, state, p = data['medium'], data['state'], data['protocol']
    branch = chr(65+data['branch'])
    fig = new_figure(f'SolsBaby  /  experience {branch} changes the response')
    axes = fig.subplots(2, 2)
    image = draw_tree(axes[0, 0], medium.morphology, state.tag[0])
    colorbar = fig.colorbar(image, ax=axes[0, 0], shrink=.65, pad=.01)
    colorbar.set_label('Slow state p', color=MUTED)
    colorbar.ax.tick_params(colors=MUTED)
    axes[0, 0].text(.02, .01, 'Fast voltage cleared; slow conductance remains.',
                     transform=axes[0, 0].transAxes, color=MUTED, fontsize=9)
    draw_probe(axes[0, 1], data['learned'], data['erased'], p)
    ax = axes[1, 0]
    style_axes(ax)
    omega = np.geomspace(.01, 3, 90)
    before = medium.transfer(np.zeros(medium.morphology.n), omega)
    after = medium.transfer(state.tag[0], omega)
    for b in range(p.branches):
        ax.loglog(omega, abs(before[:, b]), '--', color=COLORS[b], alpha=.5)
        ax.loglog(omega, abs(after[:, b]), color=COLORS[b], linewidth=1.8, label=chr(65+b))
    ax.set(xlabel='Angular frequency (inverse time)', ylabel='Tip-current → soma gain',
           title='Frozen passive EQ · dashed = erased')
    ax.legend(facecolor=PANEL, edgecolor='#314157', labelcolor=TEXT, ncol=4, fontsize=9)
    ax = axes[1, 1]
    style_axes(ax)
    m = medium.morphology
    colors = [MUTED if b < 0 else COLORS[b] for b in m.branch]
    ax.bar(np.arange(m.n), state.tag[0], color=colors, width=.8)
    ax.set(xlabel='Compartment index', ylabel='Slow shunt p', ylim=(0, 1),
           title='Memory is local · the writer receives no class label')
    return fig


def save_demo(directory, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    a, b = data['learned']['trace'], data['erased']['trace']
    summary = {'seed': int(data['protocol'].seeds[0]),
               'branch': int(data['branch']), 'active_strength': data['protocol'].active_strength,
               'relative_response_change': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-15)),
               'slow_tag_max': float(data['state'].tag.max()),
               'fast_voltage_max': float(np.abs(data['state'].v).max()),
               'units': 'dimensionless', 'protocol_sha256': data['protocol'].fingerprint}
    (directory/'demo.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig = demo_figure(data)
    fig.savefig(directory/'demo.png', dpi=150, facecolor=BACKGROUND)
    return summary


def benchmark_figure(result):
    fig = new_figure(f"SolsBaby  /  held-out memory gate: {result['gate']['status']}")
    axes = fig.subplots(2, 2)
    names = list(LABELS)
    ax = axes[0, 0]
    style_axes(ax)
    means = [result['summary'][n]['mean'] for n in names]
    ax.bar(np.arange(len(names)), means, color=[COLORS[0], MUTED, MUTED, MUTED, COLORS[4]], alpha=.85)
    for i, n in enumerate(names):
        ys = [t['accuracy'][n] for t in result['trials']]
        ax.scatter(i+np.linspace(-.1, .1, len(ys)), ys, color=TEXT, s=18, zorder=3)
    ax.axhline(1/result['protocol']['branches'], color=COLORS[1], linestyle='--', linewidth=1, label='Chance')
    ax.set_xticks(np.arange(len(names)), [LABELS[n] for n in names])
    ax.set(ylabel='Held-out accuracy', ylim=(0, 1.1), title='One soma trace · dots are independent seeds')
    ax.legend(facecolor=PANEL, edgecolor='#314157', labelcolor=TEXT, fontsize=9)
    ax = axes[0, 1]
    style_axes(ax)
    confusion = np.sum([t['confusion']['learned_soma'] for t in result['trials']], axis=0)
    normalized = confusion/np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                    color='white' if normalized[i, j] > .5 else '#172333')
    letters = [chr(65+i) for i in range(len(confusion))]
    ax.set_xticks(range(len(letters)), letters)
    ax.set_yticks(range(len(letters)), letters)
    ax.set(xlabel='Predicted experience', ylabel='Actual experience', title='Learned soma · pooled test confusion')
    ax.grid(False)
    ax = axes[1, 0]
    style_axes(ax)
    seeds = [t['seed'] for t in result['trials']]
    effect = [t['memory_effect_rms'] for t in result['trials']]
    ax.bar(range(len(seeds)), effect, color=COLORS[0])
    ax.axhline(result['protocol']['noise_std'], color=COLORS[1], linestyle='--', label='Observation noise σ')
    ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set(xlabel='Morphology / episode seed', ylabel='Raw soma voltage RMS', title='Effect size before reader standardization')
    ax.legend(facecolor=PANEL, edgecolor='#314157', labelcolor=TEXT, fontsize=9)
    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    p, b = result['protocol'], result['budget']
    lines = [
        'What this experiment measures', '',
        f"{b['compartments']} compartments · {b['persistent_tag_bytes']} bytes of slow state",
        f"{b['somatic_ports']} soma port · {b['somatic_samples_per_read']} samples per probe",
        f"{p['train_per_class']*p['branches']} train / {p['test_per_class']*p['branches']} test episodes per seed",
        'Fast voltage is reset before the probe.',
        'Writing is frozen during measurement; decay continues.', '',
        f"This is an engineered {p['branches']}-context memory demonstration.",
        f"The context label itself needs only {b['class_label_information_bits']:g} bits.",
        'No storage, speed, or biological superiority is established.', '',
        f"Protocol: {result['protocol_sha256'][:24]}…",
    ]
    ax.text(.04, .96, '\n'.join(lines), transform=ax.transAxes, va='top', color=TEXT, fontsize=10, linespacing=1.6)
    return fig


def save_benchmark(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (directory/'protocol.json').write_text(json.dumps(result['protocol'], indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (directory/'episodes.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'condition', 'test_episode', 'actual', 'predicted', 'correct'])
        for trial in result['trials']:
            for name, predictions in trial['predictions'].items():
                for i, (actual, predicted) in enumerate(zip(trial['test_labels'], predictions)):
                    writer.writerow([trial['seed'], name, i, actual, predicted, int(actual == predicted)])
    fig = benchmark_figure(result)
    fig.savefig(directory/'report.png', dpi=150, facecolor=BACKGROUND)
