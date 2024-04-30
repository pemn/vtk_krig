#!python
# create a regular vtk grid from a schema and bounding box
# schema: block sizes, and optionally offset and rotation. ex.: 10
# xyz0: bottom left point of the grid bounding box
# xyz1: upper right point of the grid bounding box
# border: create a border of x cells around the bounding box
# ijk: calculate block index field
# i_j_k: create three indexes, one for each dimention
# n: create a increasing variable that goes from 0 to 1 across all blocks
# row: create a increasing variable that goes from 1 to n of blocks
# volume: calculate volume field

'''
usage: $0 schema x0 y0 z0 x1 y1 z1 border=0,1 ijk@ i_j_k@ n@ row@ volume@ output*vtk,csv,xlsx display@
'''
import sys, os.path, re
import numpy as np
import pandas as pd


# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, log

from pd_vtk import vtk_Voxel, pv_save, vtk_mesh_info, vtk_grid_flag_ijk, vtk_shape_ijk, vtk_plot_grids

def vtk_create_grid(schema, x0, y0, z0, x1, y1, z1, border, ijk, i_j_k, n, row, volume, output, display):
  if not border:
    border = 0
  else:
    border = float(border)
  bb = np.array(((x0, y0, z0), (x1, y1, z1)), np.float_)
  grid = vtk_Voxel.from_bb_schema(bb, schema, 3, border)
  if int(ijk):
    vtk_grid_flag_ijk(grid, 'ijk')

  if int(i_j_k):
    s = vtk_shape_ijk(grid.dimensions, True)
    grid.cell_data['i'] = np.ravel(np.broadcast_to(np.arange(s[2]), s))
    grid.cell_data['j'] = np.ravel(np.transpose(np.broadcast_to(np.arange(s[1]), np.roll(s, 1)), (1,2,0)))
    grid.cell_data['k'] = np.ravel(np.rollaxis(np.broadcast_to(np.arange(s[0]), np.roll(s, 2)), 2))
    
  if int(n):
    grid['n'] = np.linspace(1, 0, grid.n_cells, True)

  if int(row):
    grid['row'] = np.arange(1, grid.n_cells + 1)

  if int(volume):
    grid.cells_volume('volume')


  print(vtk_mesh_info(grid))
  if output:
    pv_save(grid, output)

  if int(display):
    vtk_plot_grids(grid, grid.active_scalars_name)

  log("# vtk_create_grid finished")

main = vtk_create_grid

if __name__=="__main__":
  usage_gui(__doc__)
