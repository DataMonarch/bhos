import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.cm as cm


def distance(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5

import dis
def scale(data):
    """
    data (list, numpy array): the array to be normalized
    Returns: weights based on the squared inverse of distance scaled to sum up to 1
    """
    if isinstance(data, list):
        data = np.array(data)

    return (1 / data)**6 / ((1 / data)**6).sum()

def value_mapping(point, well_loc, well_property, n_neighbors=None, weight_cutoff = 0.2):
    distances = np.array([distance(point, loc) for loc in well_loc])
    idx = np.argsort(distances)
    distances = distances[idx]
    well_property = well_property[idx]

    if n_neighbors:
        if n_neighbors > len(well_property):
            raise Exception('Number of reference neighbors cannot exceed the number of wells!')
        distances = distances[:n_neighbors]
        well_property = well_property[:n_neighbors]

    if weight_cutoff:
        scaled = scale(distances)
        weights = scaled[scaled >= weight_cutoff]
        well_property = well_property[:len(weights)]

    else:
        weights = scale(distances)

    return np.dot(weights, well_property)

# property_mapping(p1, well_loc, well_phi)

top_map = np.asarray(Image.open('top_map.png'))

# top_map.shape
well_loc = [[1224, 1155],
            [1150, 1317],
            [1525, 1286],
            [923, 791],
            [1432, 808],
            [1213, 873]]

well_phi = np.array([0.22, 0.16, 0.25, 0.23, 0.24, 0.27])
well_sw = np.array([0.23, 0.43, 0.29, 0.20, 0.44, 0.39])
well_th = np.array([550, 229, 447, 505, 304, 457])


properties = [well_phi, well_sw, well_th]
# property_map.shape

def map_properties(dims_map, properties, well_loc, grid_size, n_neighbors=None, weight_cutoff=0.2):

    property_map = np.array([np.empty(dims_map) for property in properties])

    grid_x = grid_size[0]
    grid_y = grid_size[1]


    for i in range(0, dims_map[0], grid_y):
        for j in range(0, dims_map[1], grid_x):
            # print([i, j])
            # distances.append(scale([distance([j, i], loc) for loc in well_loc]))
            if [j, i] not in well_loc:
                if j+grid_x > dims_map[1] and i+grid_y < dims_map[0]:

                    for idx, property in enumerate(properties):
                        property_map[idx, i : i+grid_y, j : ] = value_mapping([j, i+grid_y], well_loc, property, n_neighbors, weight_cutoff)

                    # phi_map[i : i+grid_y, j : ] = value_mapping([j, i+grid_y], well_loc, well_phi, n_neighbors)
                    # sw_map[i : i+grid_y, j : ] = value_mapping([j, i+grid_y], well_loc, well_sw, n_neighbors)
                    # th_map[i : i+grid_y, j : ] = value_mapping([j, i+grid_y], well_loc, well_th, n_neighbors)



                elif j+grid_x < dims_map[1] and i+grid_y > dims_map[0]:
                    for idx, property in enumerate(properties):
                        property_map[idx, i:, j : j+grid_x] = value_mapping([j+grid_x, i], well_loc, property, n_neighbors, weight_cutoff)

                    # phi_map[i:, j : j+grid_x] = value_mapping([j+grid_x, i], well_loc, well_phi, n_neighbors)
                    # sw_map[i:, j : j+grid_x] = value_mapping([j+grid_x, i], well_loc, well_sw, n_neighbors)
                    # th_map[i:, j : j+grid_x] = value_mapping([j+grid_x, i], well_loc, well_th, n_neighbors)

                else:
                    for idx, property in enumerate(properties):
                        property_map[idx, i : i+grid_y, j : j+grid_x] = value_mapping([j+grid_x, i+grid_y], well_loc, property, n_neighbors, weight_cutoff)

                    # phi_map[i : i+grid_y, j : j+grid_x] = value_mapping([j+grid_x, i+grid_y], well_loc, well_phi, n_neighbors)
                    # sw_map[i : i+grid_y, j : j+grid_x] = value_mapping([j+grid_x, i+grid_y], well_loc, well_sw, n_neighbors)
                    # th_map[i : i+grid_y, j : j+grid_x] = value_mapping([j+grid_x, i+grid_y], well_loc, well_th, n_neighbors)
            else:
                print('Well encountered at ', [i, j])
                for idx, property in enumerate(properties):
                        property_map[idx, i, j] = property[np.where(np.array(well_loc) == j)[0]]

                # phi_map[i, j] = well_phi[np.where(np.array(well_loc) == j)[0]]
                # sw_map[i, j] = well_sw[np.where(np.array(well_loc) == j)[0]]
                # th_map[i, j] = well_th[np.where(np.array(well_loc) == j)[0]]

    for i, y, x in zip(range(len(well_loc)), np.array(well_loc)[:, 0], np.array(well_loc)[:, -1]):
        # phi_map[y, x] = well_phi[i]
        # sw_map[y, x] = well_sw[i]

        for idx, property in enumerate(properties):
            property_map[idx, y, x] = property[i]

    return property_map



property_maps = map_properties(top_map.shape[:-1], properties, well_loc, [10, 10], n_neighbors=None, weight_cutoff=0)

phi_map = property_maps[0] * 100
sw_map = property_maps[1] * 100
th_map = property_maps[2]

plt.hist(np.array(distances).flatten())
print(np.where(property_maps[0] == 1))


#
#
# property_map = np.array([[property_mapping([i, j], well_loc, well_phi) for i in range(top_map.shape[1])] for j in range(top_map.shape[0])])
# property_map.shape
# sw_map = np.array([[property_mapping([i, j], well_loc, well_sw) for i in range(top_map.shape[1])] for j in range(top_map.shape[0])])


wells_info = pd.DataFrame(np.array(well_loc), columns=['well_loc_x', 'well_loc_y'])
wells_info['phi'] = well_phi
wells_info['sw'] = well_sw
wells_info['th'] = well_th

%matplotlib qt
# plt.hist(property_maps[0][property_maps[0]>0].flatten())

plt.imshow(phi_map, alpha=1, cmap='turbo')
plt.colorbar()
plt.imshow(top_map)
plt.plot(wells_info['well_loc_x'], wells_info['well_loc_y'], 'wo', linewidth=2)
plt.title('Porosity Map (%)')
# plt.title('Water Saturation Map (%)')
