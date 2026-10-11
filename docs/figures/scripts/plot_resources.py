"""Measured resource ablation, adapted from paper-plot grouped-bar templates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'evidence/resource-comparison/summary.json'
OUT = ROOT / 'docs/figures'
NAMES = {'iq2_xs':'IQ2_XS', 'swift':'Swift IQ3_XXS', 'iq3_s':'IQ3_S'}
COLORS = {'default':'#A6AAA5', 'shared':'#AB947B', 'incumbent':'#426F69'}
LABELS = {'default':'Upstream resource defaults', 'shared':'Upstream + shared tuning',
          'incumbent':'Turbo preset'}


def main():
    data = json.loads(DATA.read_text())
    assert data['complete']
    counts = sorted({r['count'] for r in data['rows']})
    plt.style.use(Path(__file__).with_name('readme.mplstyle'))
    fig, axes = plt.subplots(1,len(counts),figsize=(8.7*len(counts),4.25),squeeze=False)
    for axis,count in zip(axes[0],counts):
        rows = {r['weight']:r for r in data['rows'] if r['count']==count}
        x = np.arange(3)
        maximum = 0
        for index,arm in enumerate(COLORS):
            values = [rows[w]['arms'][arm]['median_e2e_tps'] for w in NAMES]
            lower = [v-rows[w]['arms'][arm]['minimum_e2e_tps'] for w,v in zip(NAMES,values)]
            upper = [rows[w]['arms'][arm]['maximum_e2e_tps']-v for w,v in zip(NAMES,values)]
            maximum = max(maximum,max(v+u for v,u in zip(values,upper)))
            bars = axis.bar(x+(index-1)*.25, values,width=.235,color=COLORS[arm],
                            hatch='//' if arm=='incumbent' else None,edgecolor='white',
                            label=LABELS[arm],yerr=[lower,upper],
                            error_kw={'ecolor':'#58635C','elinewidth':.8,'capsize':2.5})
            for bar,v,u in zip(bars,values,upper):
                axis.text(bar.get_x()+bar.get_width()/2,v+u+2.0,f'{v:.1f}',ha='center',
                          va='bottom',fontsize=9,fontweight='bold' if arm=='incumbent' else 'normal')
        axis.set_xticks(x,labels=list(NAMES.values()))
        axis.set_ylim(0,np.ceil((maximum+20)/25)*25)
        axis.set_ylabel('Aggregate output tokens / second')
        axis.set_title(f'{count} simultaneous requests',loc='center',fontsize=11)
        axis.grid(axis='x',visible=False)
    fig.suptitle('Resource tuning and batch MTP: separate contributions',fontsize=13,y=.98)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,fontsize=9)
    fig.text(.5,.025,'Same pinned numerical engine and weights; 2 independent starts per arm; observed range; native cap 262,144\n256 output tokens per request; 3 mixed-task batches per start; includes prefill/admission; thinking/cache off',
             ha='center',fontsize=8.6,color='#606964')
    fig.tight_layout(rect=(0,.12,1,.80),pad=.8)
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'resource-comparison.{suffix}',dpi=300,facecolor='white')
    svg = OUT/'resource-comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    metadata=dict(source_data=str(DATA.relative_to(ROOT)),source_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  script=str(Path(__file__).relative_to(ROOT)),
                  exports=['docs/figures/resource-comparison.svg','docs/figures/resource-comparison.png'],
                  surface_class='connector_milestone',comparison='Explicit resource-policy ablation and remaining batch-MTP effect; every measured model/slot point is retained.',
                  template_provenance='paper-plot bar_grouped_hatch; figure-polish muted academic style',
                  created_utc=datetime.now(timezone.utc).isoformat(),visually_reviewed=False)
    (OUT/'resource-comparison-catalog.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Rendered measured resource comparison; visual inspection required.')


if __name__=='__main__':
    main()
