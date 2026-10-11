"""README charts adapted from paper-plot's line_aime.py and grouped-bar templates.

Input: evidence/showcase/benchmark-summary.json. No synthetic measurements.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/figures'
DATA = ROOT / 'evidence/showcase/benchmark-summary.json'
STYLE = Path(__file__).with_name('readme.mplstyle')
COLORS = {'official': '#9A9A94', 'turbo': '#426F69'}
NAMES = {'iq2_xs': 'IQ2_XS', 'swift': 'Swift IQ3_XXS', 'iq3_s': 'IQ3_S'}
TASKS = {'english': 'English prose', 'code': 'Python code', 'chinese': 'Chinese prose'}
LEGEND = [Line2D([], [], color=COLORS['official'], marker='s', linestyle='--', label='Upstream (shared tuning)'),
          Line2D([], [], color=COLORS['turbo'], marker='o', label='Strata-Turbo-Single')]


def export(fig, name, entries, claim):
    fig.savefig(OUT / (name + '.png'), dpi=300, facecolor='white')
    svg = OUT / (name + '.svg')
    fig.savefig(svg, facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    entries.append({'id': name, 'source_data': str(DATA.relative_to(ROOT)),
                    'script': str(Path(__file__).relative_to(ROOT)),
                    'exports': [f'docs/figures/{name}.svg', f'docs/figures/{name}.png'],
                    'surface_class': 'connector_milestone', 'comparison': claim,
                    'visually_reviewed': False})


def curves(data, weight, entries):
    if any(weight + '-' + arm not in data['curves'] for arm in COLORS):
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.45), sharey=True)
    maximum = max(p['maximum_decode_tps'] for arm in COLORS for p in data['curves'][weight+'-'+arm]['points'])
    for axis, (family, title) in zip(axes, TASKS.items()):
        for arm in COLORS:
            points = [p for p in data['curves'][weight+'-'+arm]['points'] if p['family'] == family]
            x = np.array([p['prompt_tokens'] for p in points])
            y = np.array([p['median_decode_tps'] for p in points])
            axis.plot(x, y, color=COLORS[arm], marker='o' if arm=='turbo' else 's',
                      linestyle='-' if arm=='turbo' else '--', linewidth=2.2 if arm=='turbo' else 1.7)
            axis.fill_between(x, [p['minimum_decode_tps'] for p in points],
                              [p['maximum_decode_tps'] for p in points], color=COLORS[arm], alpha=0.11)
        axis.set_xscale('log', base=2)
        axis.set_xticks([100, 8192, 65536, 254000], labels=['~100', '8k', '64k', '254k'])
        axis.set_xlim(65, 380000)
        axis.set_ylim(0, np.ceil(maximum/25)*25+10)
        axis.set_title(title, loc='center', fontsize=11)
        axis.set_xlabel('Input tokens (log scale)', fontsize=9.5)
        axis.grid(axis='x', visible=False)
    axes[0].set_ylabel('Decode tokens / second')
    fig.suptitle(f'{NAMES[weight]}: output speed across input lengths', fontsize=13, y=0.98)
    fig.legend(handles=LEGEND, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=2)
    fig.text(0.5, 0.025, '512 output tokens; native cap 262,144; median and observed range of 2 outputs; prefill excluded',
             ha='center', fontsize=8.4, color='#606964')
    fig.tight_layout(rect=(0, 0.07, 1, 0.82), pad=0.7, w_pad=0.9)
    export(fig, 'decode-'+weight, entries, 'Paired output-stage speed by task and actual input length; cache policy documented separately.')


def prefill(data, entries):
    if len(data['curves']) != 6:
        return
    fig, axis = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(3)
    for index, arm in enumerate(COLORS):
        values = [next(p['prefill_tps']/1000 for p in data['curves'][weight+'-'+arm]['cold_prefill']
                       if p['background_tokens']==253952) for weight in NAMES]
        bars = axis.bar(x+(index-.5)*.32, values, width=.32, color=COLORS[arm],
                        edgecolor='white', hatch='//' if arm=='turbo' else None,
                        label='Strata-Turbo-Single' if arm=='turbo' else 'Upstream (shared tuning)')
        for bar, value in zip(bars, values):
            axis.text(bar.get_x()+bar.get_width()/2, value+.065, f'{value:.2f}k',
                      ha='center', va='bottom', fontsize=9, fontweight='bold' if arm=='turbo' else 'normal')
    axis.set_xticks(x, labels=list(NAMES.values()))
    axis.set_ylim(0, 5.1)
    axis.set_ylabel('Prefill throughput (thousand tokens / second)')
    axis.set_title('Cold long-input prefill: all six configurations', fontsize=12)
    axis.grid(axis='x', visible=False)
    axis.legend(loc='upper right', ncol=2, fontsize=8.5)
    fig.text(.5,.025,'~254k input; zero cached tokens; one observation per configuration; native cap 262,144',
             ha='center',fontsize=8.3,color='#606964')
    fig.tight_layout(rect=(0,.07,1,1),pad=.9)
    export(fig,'prefill-six-configurations',entries,'Cold input-stage throughput across three weights and two release presets.')


def concurrency(data, entries):
    if len(data['concurrency']) != 6 or any(len(rows)!=4 for rows in data['concurrency'].values()):
        return
    fig, axes = plt.subplots(1,3,figsize=(9.5,3.45),sharey=True)
    maximum=max(r['aggregate_e2e_tps'] for rows in data['concurrency'].values() for r in rows)
    for axis,(weight,title) in zip(axes,NAMES.items()):
        for arm in COLORS:
            rows=data['concurrency'][weight+'-'+arm]
            axis.plot([r['count'] for r in rows],[r['aggregate_e2e_tps'] for r in rows],
                      color=COLORS[arm],marker='o' if arm=='turbo' else 's',linestyle='-' if arm=='turbo' else '--')
        axis.set_xticks([1,2,4,6])
        axis.set_ylim(0,np.ceil(maximum/25)*25+10)
        axis.set_xlabel('Concurrent requests')
        axis.set_title(title,loc='center',fontsize=11)
        axis.grid(axis='x',visible=False)
    axes[0].set_ylabel('Aggregate tokens / second')
    fig.suptitle('Concurrent throughput with mixed tasks',fontsize=13,y=.98)
    fig.legend(handles=LEGEND,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2)
    fig.text(.5,.025,'256 output tokens per request; 3 batches; includes prefill and admission; prompt cache off',
             ha='center',fontsize=8.4,color='#606964')
    fig.tight_layout(rect=(0,.07,1,.82),pad=.7,w_pad=.9)
    export(fig,'concurrent-throughput',entries,'Paired aggregate end-to-end throughput for 1, 2, 4 and 6 simultaneous mixed-task requests.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    data=json.loads(DATA.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    plt.style.use(STYLE)
    plt.rcParams.update({'svg.fonttype':'none','font.size':10,'hatch.linewidth':0.6})
    entries=[]
    for weight in NAMES:curves(data,weight,entries)
    prefill(data,entries)
    concurrency(data,entries)
    (OUT/'catalog.json').write_text(json.dumps({'data_complete':data['complete'],
        'created_utc':datetime.now(timezone.utc).isoformat(),'matplotlib':matplotlib.__version__,
        'source_numeric_sha256':hashlib.sha256(json.dumps({k:data[k] for k in ('curves','concurrency')},sort_keys=True).encode()).hexdigest(),
        'template_provenance':'paper-plot line_training_curve and bar_grouped_hatch; figure-polish academic style',
        'figures':entries},indent=2)+'\n')
    print('Rendered',len(entries),'measured figures; visual review required.')


if __name__=='__main__':main()
