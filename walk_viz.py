import sys
import matplotlib.pyplot as plt
import seaborn


def parse():
    coords = []
    coord  = []
    for line in sys.stdin:
        if '-\n' in line:
            coords.append(coord)
            coord = []
        else:
            coord.append(int(line))

    coords.append(coord)
    return coords


def sqrt_dist(coord1, coord2):
    assert len(coord1) == len(coord2)
    
    out = 0
    for c1, c2 in zip(coord1, coord2):
        out += (c1 - c2)**2

    return out


def valid_walk(coords):
    for i in range(1, len(coords)):
        if sqrt_dist(coords[i - 1], coords[i]) != 1:
            return False

    return True


def draw(coords):
    seaborn.set(style='ticks')
    fig = plt.figure()
    if len(coords[0]) == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    seaborn.despine(ax=ax, offset=0)
    
    ax.plot(*zip(*coords),         linestyle='-', color='b')
    ax.plot(*zip(*(coords[1:-1])), linestyle='',  color='b', marker='o', markersize=2)
    if coords[0] == coords[-1]:
        ax.scatter(*zip(coords[0]), marker='x', color='r')
    else:
        ax.scatter(*zip(coords[0], coords[-1]), marker='x', color='r')

    plt.savefig("walk.pdf")
    print("Generated walk.pdf")


if __name__ == '__main__':

    coords = parse()
    if not valid_walk(coords):
        print("ERROR: invalid walk")
        exit(1)

    if len(coords) == 0:
        print("Can't draw empty walk")
    elif len(coords[0]) > 3:
        print("Can't draw walk in more than 3 dimensions")
    else:
        draw(coords)
