import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation

from evaluate import GetGraph

# Set up formatting for the movie files
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)



    

if __name__ == "__main__":
    # Get graph
    g = GetGraph('../../data/tsplib/berlin52.tsp', False)
    pos = nx.get_node_attributes(g, 'pos')
    # Build partial tours from record file.
    with open("test_rec_file.txt") as f:
        lines = [line.rstrip('\n').split() for line in f]

    partial_tours = np.array([[int(n) for n in line[1:]] for line in lines])
    edges = [[tuple(p[ii:ii+2]) for ii in range(len(p)-1)] for p in partial_tours]
    [e.append((e[-1][1],e[0][0])) for e in edges[1:]] 
    edges[0].append((partial_tours[0][0],partial_tours[0][0])) 
    # edgelist = [(str(edge[0][0]), str(edge[0][1])) for edge in edges] 

    # Build plot
    fig, ax = plt.subplots(figsize=(10,10), )
    ax.set_xlim([0, 2000])
    ax.set_ylim([0, 2000])

    def update(num):
        ax.clear()

        # Background nodes
        nx.draw_networkx(g, node_color='grey', pos=pos)
        path = partial_tours[num]
        # nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="gray")
        null_nodes = nx.draw_networkx_nodes(g, pos=pos, nodelist=set(g.nodes()) - set(path), node_color="white",  ax=ax)
        null_nodes.set_edgecolor("black")

        # Query nodes
        query_nodes = nx.draw_networkx_nodes(g, pos=pos, nodelist=path, node_color='red', ax=ax)
        query_nodes.set_edgecolor("blue")
        # nx.draw_networkx_labels(g, pos=pos, labels=dict(zip(path,path)),  font_color="white", ax=ax)
        edgelist = [(str(edge[0]), str(edge[1])) for edge in edges[num]]
        edgelist = edges[num]
        # [path[k:k+2] for k in range(len(path) - 1)]
        nx.draw_networkx_edges(g, pos=pos, edgelist=edgelist, width=2, ax=ax)

        p = [str(node) for node in path]
        # Scale plot ax
        ax.set_title("Frame %d:    "%(num+1) +  " - ".join(p), fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])


    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True


    fig.canvas.mpl_connect('button_press_event', onClick)

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=len(partial_tours), interval=1000, repeat=True)
    anim.save('berlin52_tsp_anim.mp4', writer=writer)
    plt.show()

    
    # for idx,p in enumerate(partial_tours):
    #     e = edges[idx]
    #     # c = colors
    #     c = ['r' if n in p else 'b' for n in range(len(g.nodes))]
    #     im = nx.draw_networkx(g, edgelist=e, colors=c, pos=nx.get_node_attributes(g, 'pos'))
    #     ims.append([im])
    #     fig2.clear()

    # fig3 = plt.figure(3)
    # ani = animation.ArtistAnimation(fig3, ims, interval=50, blit=True, repeat_delay=1000)

    # ani.save('dynamic_images.mp4')

    # plt.show()
    print "by"