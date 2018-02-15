from matplotlib import pyplot

def plot_frame ( frame ):
    pyplot.figure()
    pyplot.imshow(frame, cmap='Greys')
    ax = pyplot.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    pyplot.show()

def plot( sample , l = None):
    if l is None:
        l = len (sample)
    sample = array(sample)
    sample = sample.reshape(sample.shape[0], sample.shape[1], sample.shape[2], 1)
    pyplot.figure()
    for i in range(len(sample)):
        # create a gray scale subplot for each frame
        pyplot.subplot(1, l, i + 1)
        pyplot.imshow(sample[i,:,:,0], cmap='Greys')
        # turn of the scale to make it clearer
        ax = pyplot.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # show the plot
    pyplot.show()

def plot_big(sample):
    for i in range( len(sample) // 5 + 1):
        plot(sample[5 * i:min(len(sample), 5 * i + 5)], l = 5)