import pylab as pl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import collections as mc

def plot (object_data, from_frame, to_frame):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-2, 2, 0.1))
    ax.set_yticks(np.arange(-2, 2, 0.1))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    fig.set_size_inches(20, 12)
    
    for object_name in object_data:
        data = object_data[object_name]
        
        for frameNo in data:
            if from_frame  <= frameNo <= to_frame :
                # Mistake
                # data[frameNo].transform.scale = data[frameNo].transform.scale / 2
                shape = data[frameNo].get_markers()
                
                lc = mc.PolyCollection([shape], edgecolors=[colors[object_name]], 
                                       facecolors=[colors[object_name]], linewidths=[2])
                ax.add_collection(lc)


    ax.autoscale()
    ax.margins(0.1)

    plt.show()

def animate (object_data, from_frame, to_frame, min_x = -.6, max_x = 1, min_y =  -.6, max_y = 1, name = "event.mp4"):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(min_x, max_x, 0.1))
    ax.set_yticks(np.arange(min_y, max_y, 0.1))
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    fig.set_size_inches(20, 12)

    ax.autoscale()
    ax.margins(0.1)
    
#     # initialization function: plot the background of each frame
#     def init():
#         ax.clear()
#         ax.set_xticks(np.arange(-2, 2, 0.1))
#         ax.set_yticks(np.arange(-2, 2, 0.1))
#         ax.set_xlim([-2, 2])
#         ax.set_ylim([-2, 2])
#         return ax,

    # animation function.  This is called sequentially
    def anim(i):
        lc = mc.PolyCollection([object_data[object_name][i + from_frame].get_markers() for object_name in object_data], 
                               edgecolors=[colors[object_name] for object_name in object_data], 
                               facecolors=[colors[object_name] for object_name in object_data], linewidths=[2,2])
        ax.clear()
        ax.set_xticks(np.arange(min_x, max_x, 0.1))
        ax.set_yticks(np.arange(min_y, max_y, 0.1))
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.add_collection(lc)
        return lc,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, anim,
                                   frames=to_frame - from_frame, interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(name, fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.show()