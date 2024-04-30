#!python
# Overview charts of 3d voxels and samples
# v1.0 2022/10 paulo.ernesto

'''
usage: $0 voxels*vtk,csv,xlsx fields#field:voxels samples*csv,xlsx output*pdf
'''

import sys, os.path
# import modules from a pyz (zip) file with same name as scripts
sys.path.append(os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, pyd_zip_extract, pd_load_dataframe, pd_detect_xyz, commalist

pyd_zip_extract()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for python 3.5
from matplotlib.widgets import Slider
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial

def df2ijk(df, xyz = None):
  if xyz is None:
    xyz = pd_detect_xyz(df)
  return tuple(len(df[_].unique()) for _ in xyz)

def df2a3d(df, field = None):
  xyz = pd_detect_xyz(df)
  ijk = df2ijk(df, xyz)
  a2d = None
  if field and field in df:
    a2d = df.sort_values(xyz)[field].values
  elif df.shape[1] > 3:
    a2d = df.iloc[:, 3].values
  else:
    a2d = np.linspace(0, 1, df.shape[0])
  return np.reshape(a2d, ijk)

def axis_voxel_create(vxa, fc, vbf, title = None, threshold = 0.5):
  vxa.clear()
  if hasattr(vxa,'voxels'):
    vxa.voxels(vbf < threshold, facecolors=fc)
  vxa.set_title(title)

def pd_voxel_view(k3d, samples = None, title = None, pdf = None, op = None):
  vbn = np.add(k3d.shape, 1)
  if k3d.dtype.num >= 17:
    # convert string arrays to integer index
    k3d = np.reshape(pd.factorize(k3d.flat)[0], k3d.shape)
  elif k3d.dtype.num == 0:
    # convert bool to int
    k3d = k3d.astype(np.int_)
  # normalize values
  vbf = k3d
  if k3d.max() > k3d.min():
    vbf =  np.maximum(np.divide(k3d - k3d.min(), k3d.max() - k3d.min()), 0.001)
  
  # colors array is a RGBA value for each cell in the data array
  fc = np.zeros(k3d.shape + (4,))
  # instead of a i,j,k loop we can use 2d array with i,j,k indices
  fci = np.moveaxis(np.indices(k3d.shape), 0, k3d.ndim).reshape((k3d.size, k3d.ndim))

  fig = plt.figure(figsize=np.multiply(plt.rcParams["figure.figsize"], 2))
  
  rows = 220
  if samples is not None:
    plt.set_cmap('Paired')
    rows = 230
  cmap = plt.get_cmap()
  
  for i in range(k3d.size):
    it = tuple(fci[i])
    fc[it] = cmap(vbf[it])

  f = np.nanmean
  if op == 'max':
    f = np.nanmax
  if op == 'min':
    f = np.nanmin
  if op == 'std':
    f = np.nanstd

  cb = None
  if title:
    fig.suptitle(title)
  plt.subplot(rows + 1)
  plt.grid(True)
  s = plt.gca().matshow(f(k3d, 2).T, cmap=cmap, origin='lower')
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.gca().set_title('z')

  plt.subplot(rows + 2)
  plt.grid(True)
  plt.gca().matshow(f(k3d, 0).T, cmap=cmap, origin='lower')
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.gca().set_title('x')
  
  plt.subplot(rows + 3)
  plt.grid(True)
  plt.gca().matshow(f(k3d, 1).T, cmap=cmap, origin='lower')
  plt.gca().set_xticklabels([])
  plt.gca().set_yticklabels([])
  plt.gca().set_title('y')

  if samples is None:
    plt.subplot(rows + 4, projection='3d')
    fig.colorbar(cb, ax=plt.gca(), location='top')
    plt.gca().axis('off')
    plt.gca().voxels(vbf, facecolors=cmap(vbf))
  else:
    plt.subplot(rows + 4)
    plt.gca().set_title('cutoff')
    sla = plt.gca()
    fig.colorbar(cb, ax=sla, location='top')
    sbb = sla.get_position()
    sbb.update_from_data_xy([sbb.p0, [sbb.x1, sbb.y0 + (sbb.y1 - sbb.y0) * 0.1]])
    vs = Slider(sla, None, 0, 1)
    sla.set_position(sbb)

    plt.subplot(rows + 5, projection='3d')
    # voxels is only available since matplotlib 2.1
    # vulcan bundled matplotlib is version 1.5.3
    axis_voxel_create(plt.gca(), fc, vbf, 'voxels')
    plt.gca().set_title('voxels')
    # the partial must use the 3d gca
    vs.on_changed(partial(axis_voxel_create, plt.gca(), fc, vbf, 'voxels'))

    plt.subplot(rows + 6, projection='3d')
    c = None
    if samples.shape[1] >= 4:
      c=samples[:, 3]
    plt.gca().scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=c, cmap=cmap if c else None)
    plt.gca().set_title('samples')

  if pdf is not None:
    pdf.savefig()
  else:
    plt.show()

def db_voxel_view(voxels, fields, samples = None, output = None):
  a_hard = None
  df_soft = None
  pdf = None
  if output:
    pdf = PdfPages(output)

  a3d = []
  vl = commalist().parse(fields).split()
  if voxels.lower().endswith('vtk'):
    from pd_vtk import pv_read, vtk_array_ijk
    mesh = pv_read(voxels)
    for v in vl:
      if v in mesh.array_names and mesh.GetDataObjectType() in [2,6]:
        # transform flat array into xyz 3d array
        #if mesh.get_array_association(v) == pv.FieldAssociation.CELL:
        #  a3d.append(np.transpose(np.reshape(mesh.get_array(v), np.maximum(np.subtract(np.flip(mesh.dimensions), 1), 1)), (2,1,0)))
        #else:
        #  a3d.append(np.transpose(np.reshape(mesh.get_array(v), np.maximum(np.flip(mesh.dimensions), 1)), (2,1,0)))
        a3d.append(vtk_array_ijk(mesh, v))
  elif voxels:
    df_soft = pd_load_dataframe(voxels)
    for field in vl:
      a3d.append(df2a3d(df_soft, field))
  if samples:
    df = pd_load_dataframe(samples)
    xyz_hard = pd_detect_xyz(df)
    a_hard = df[xyz_hard].dropna().values

  for i in range(len(a3d)):
    print(vl[i], a3d[i].shape)
    pd_voxel_view(a3d[i], a_hard, vl[i], pdf)

  if pdf is not None: 
    pdf.close()

main = db_voxel_view

if __name__ == '__main__':
  usage_gui(__doc__)
