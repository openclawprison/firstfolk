"""Export actual cortical centroid views and the explicitly synthetic routing graph."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'work'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import brain


def render(destination):
    atlas=brain.atlas()
    fig=plt.figure(figsize=(16,11),facecolor='#f7f9fc')
    grid=fig.add_gridspec(2,3,height_ratios=[1,1.2])
    colors={'Vis':'#7c3a96','SomMot':'#4682b4','DorsAttn':'#149e4f',
            'SalVentAttn':'#c54d9e','Limbic':'#ba9b15','Cont':'#ed8b24','Default':'#c73045'}
    views=[(0,1,'Axial: right / anterior'),(0,2,'Coronal: right / superior'),(1,2,'Sagittal: anterior / superior')]
    for i,(x,y,title) in enumerate(views):
        ax=fig.add_subplot(grid[0,i])
        for network,color in colors.items():
            points=[p['centroid_ras_mm'] for p in atlas['parcels'] if p['network']==network]
            ax.scatter([p[x] for p in points],[p[y] for p in points],c=color,s=32,alpha=0.85)
        ax.set_title(title,fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlabel('MNI millimetres'); ax.set_ylabel('MNI millimetres')
        ax.grid(alpha=0.15)
        ax.spines[['top','right']].set_visible(False)
    graph=fig.add_subplot(grid[1,:])
    positions={
        'sensory':(0,3),'thalamus':(1,3),'salience':(2,4),'hippocampus':(2,2),
        'semantic':(3.2,1),'working_memory':(3.5,3),'executive':(5,3),
        'valuation':(5,1),'striatum':(6.5,2),'prediction_error':(6.5,0),
        'threat':(0,4.5),'homeostasis':(0,0.5),'social':(2,0),
        'self_model':(4.5,4.5),'cerebellum':(8,0.5),'motor':(8,3)}
    for source,target in brain.EDGES:
        graph.annotate('',xy=positions[target],xytext=positions[source],
                       arrowprops={'arrowstyle':'->','color':'#a4afc0','alpha':0.6,'connectionstyle':'arc3,rad=0.08'})
    for key,(x,y) in positions.items():
        graph.text(x,y,key.replace('_','\n'),ha='center',va='center',fontsize=9,
                   bbox={'boxstyle':'round,pad=0.55','facecolor':'#e5ecf6','edgecolor':'#546b8a'})
    graph.set_xlim(-0.7,8.7);graph.set_ylim(-0.6,5.1);graph.axis('off')
    graph.set_title('Backend functional graph: 16 modules / 24 hand-designed routes',fontsize=13,pad=18)
    handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=color,label=name,markersize=8) for name,color in colors.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,0.92),ncol=7,frameon=False)
    fig.suptitle('Brain map: published atlas metadata + synthetic cognitive architecture',fontsize=17,y=0.98)
    fig.text(0.5,0.025,'Top: Schaefer et al. (2018), 100 cortical parcel centroids / 7 networks. No cortical mesh or connectome.\n'
             'Bottom: software modules inspired by research; arrows and activity are not measured human neural connections.',
             ha='center',fontsize=10,color='#485269')
    fig.subplots_adjust(top=0.86,bottom=0.1,hspace=0.45,wspace=0.3)
    fig.savefig(destination,dpi=150)
    plt.close(fig)


if __name__=='__main__':
    render(Path(__file__).parent/'research'/'brain_map.png')
