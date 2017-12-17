import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import collections as mc
from utils import SESSION_NAME, SESSION_OBJECTS, SESSION_EVENTS, SESSION_LEN, SESSION_OBJ_2D, START, END
import io
import base64
from IPython.display import HTML, display_html

colors = [ (1, 0, 0, 1), (0,1,0,1), (0,0,1,1), 
          (0.5, 0.5, 0, 1), (0,0.5, 0.5,1), (0.5, 0, 0.5,1),
         (0.7, 0.3, 0, 1), (0,0.7, 0.3,1), (0.7, 0, 0.3,1),
         (0.3, 0.7, 0, 1), (0,0.3, 0.7,1), (0.3, 0, 0.7,1)]

def plot (session, from_frame, to_frame, show = True):
    """

    """
    object_data = session[SESSION_OBJ_2D]
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-2, 2, 0.1))
    ax.set_yticks(np.arange(-2, 2, 0.1))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    fig.set_size_inches(20, 12)
    
    color_counter = 0
    for object_name in object_data:
        data = object_data[object_name]
        
        for frameNo in data:
            if from_frame  <= frameNo <= to_frame :
                # Mistake
                # data[frameNo].transform.scale = data[frameNo].transform.scale / 2
                shape = data[frameNo].get_markers()
                
                lc = mc.PolyCollection([shape], edgecolors=[colors[color_counter]], 
                                       facecolors=[colors[color_counter]], linewidths=[2])
                ax.add_collection(lc)

        color_counter += 1
    ax.autoscale()
    ax.margins(0.1)

    if show:
        plt.show()

def animate (session, from_frame, to_frame, min_x = -.6, max_x = 1, min_y =  -.6, max_y = 1, 
                name = "temp.mp4'", show = True, colors = colors):
    """
    
    """
    object_data = session[SESSION_OBJ_2D]
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(min_x, max_x, 0.1))
    ax.set_yticks(np.arange(min_y, max_y, 0.1))
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    fig.set_size_inches(6, 6)

    ax.autoscale()
    ax.margins(0.1)

    # animation function.  This is called sequentially
    def anim(i):
        lc = mc.PolyCollection([object_data[object_name][i + from_frame].get_markers() for object_name in object_data], 
                               edgecolors=[colors[j] for j in range(len(object_data))], 
                               facecolors=[colors[j] for j in range(len(object_data))], linewidths=[2,2])
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
    if name:
        anim.save(name, fps=30, extra_args=['-vcodec', 'libx264'])
    
        if show:
            video = io.open(name, 'r+b').read()
            encoded = base64.b64encode(video)
            display_html(HTML(data='''<video alt="test" controls>
                            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                         </video>'''.format(encoded.decode('ascii'))))

def animate_event(session, event_index, min_x = -.6, max_x = 1, min_y =  -.6, max_y = 1, 
                name = 'temp.mp4', show = True, colors = colors ):
    """
    Visualize an event in the session, given the event_index
    """
    if event_index < 0 or event_index >= len(session[SESSION_EVENTS]):
        return

    event = session[SESSION_EVENTS][event_index]

    start = event[START]
    end = event[END]

    animate(session, start, end, min_x, max_x, min_y, max_y, name, show, colors)