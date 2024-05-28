#!python
# vtk grid data krigging using pykrige
# soft_db: grid to be estimated
# hard_db: samples with hard data
# lito: (optional) run a distinct pass for each lito value
# variables: select which variables from hard data will be interpolated in the soft data
# variogram: (optional) a json or yaml file with variogram parameters
# v1.0 2024/04 paulo.ernesto
# ! pip install pykrige
'''
usage: $0 soft_db*vtk hard_db*csv,xlsx,bmf,vtk lito:hard_db variables#variable:hard_db variogram*yaml,json output*vtk display@
'''

import sys, os.path
import numpy as np
import pandas as pd

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

import pyvista as pv

from _gui import usage_gui, log, commalist, pd_load_dataframe, pd_detect_xyz

default_variogram = dict(
algorithm='ordinary',
variogram_model= 'gaussian',
variogram_parameters=None,
nlags=6,
anisotropy_scaling_y=1.0,
anisotropy_scaling_z=1.0,
anisotropy_angle_x=0.0,
anisotropy_angle_y=0.0,
anisotropy_angle_z=0.0
)

class KrigVar(dict):
  def __init__(self, fp = None):
    self.update(default_variogram)
    if not fp:
      ...
    elif isinstance(fp, dict):
      self.update(fp)
    elif not os.path.exists(fp):
      print("file not found:", fp)
    elif result.lower().endswith('json'):
      import json
      self.update(json.load(open(fp, 'r')))
    elif result.lower().endswith('yaml'):
      import yaml
      self.update(yaml.safe_load(open(fp, 'r')))

  def get_kw(self, *args):
    return dict(filter(lambda _: _[0] in args, self.items()))

  def krig3d(self, samples, points):
    # Create the 3D ordinary kriging object and solves for the three-dimension kriged
    ka = None
    k3d = None
    kw = self.get_kw('variogram_model','variogram_parameters','nlags','anisotropy_scaling_y','anisotropy_scaling_z','anisotropy_angle_x','anisotropy_angle_y','anisotropy_angle_z')
    if self.get('algorithm') == 'ordinary':
      from pykrige.ok3d import OrdinaryKriging3D
      # OrdinaryKriging3D(x, y, z, val, variogram_model='linear', variogram_parameters=None, variogram_function=None, nlags=6, weight=False, anisotropy_scaling_y=1.0, anisotropy_scaling_z=1.0, anisotropy_angle_x=0.0, anisotropy_angle_y=0.0, anisotropy_angle_z=0.0, 
      # verbose=False, enable_plotting=False, exact_values=True, pseudo_inv=False, pseudo_inv_type='pinv')
      ka = OrdinaryKriging3D(samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], **kw)
    if self.get('algorithm') == 'universal':
      from pykrige.uk3d import UniversalKriging3D
      # UniversalKriging3D(x, y, z, val, variogram_model='linear', variogram_parameters=None, variogram_function=None, nlags=6, weight=False, anisotropy_scaling_y=1.0, anisotropy_scaling_z=1.0, anisotropy_angle_x=0.0, anisotropy_angle_y=0.0, anisotropy_angle_z=0.0,
      # drift_terms=None, specified_drift=None, functional_drift=None, 
      # verbose=False, enable_plotting=False, exact_values=True, pseudo_inv=False, pseudo_inv_type='pinv')
      ka = UniversalKriging3D(samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], **kw)
    
    if ka is not None:
      k3d, ss3d = ka.execute('points', points[:, 0], points[:, 1], points[:, 2])
      print(pd.Series(k3d).describe())
      ka.print_statistics()
    
    return k3d

  def __call__(self, grid, df, lito, vl):
    xyz = pd_detect_xyz(df)
    hard_lito = np.full(df.shape[0], None)
    points = grid.cell_centers().points
    soft_lito = np.full(points.shape[0], None)
    full_lito = [None]
    if lito and lito in grid.array_names and lito in df:
      soft_lito = grid.get_array(lito, 'cell')
      hard_lito = df[lito]
      full_lito = set(hard_lito).intersection(soft_lito)

    for l in full_lito:
      soft_bi = np.equal(soft_lito, l)
      hard_bi = np.equal(hard_lito, l)
      for v in vl:
        log("lito", l, "variable", v)
        s = df.loc[hard_bi, xyz + [v]].dropna().values
        d = np.full(soft_bi.size, np.nan)
        if s.size:
          r = self.krig3d(s, points[soft_bi])
          if r is not None:
            d[soft_bi] = r
        grid.cell_data[v] = d


def main(soft_db, hard_db, lito, variables, variogram, output, display):
  log("# vtk_krig started")
  grid = pv.read(soft_db)

  kv = KrigVar(variogram)
  vl = variables.split(';')
  kv(grid, pd_load_dataframe(hard_db), lito, vl)
  if output:
    grid.save(output)

  if int(display):
    from pd_vtk import vtk_voxel_view, vtk_mesh_info
    for v in vl:
      vtk_voxel_view(grid, v)

  log("finished")


if __name__=="__main__":
  usage_gui(__doc__)
