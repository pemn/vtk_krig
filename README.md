## 📌 Description
vtk grid data krigging using pykrige
## 📸 Screenshot
![screenshot1](https://github.com/pemn/assets/blob/main/vtk_krig1.png?raw=true)
## 📝 Parameters
name|optional|description
---|---|------
soft_db|❎|vtk grid to be estimated
hard_db|❎|samples with hard data
lito|☑️|run a distinct pass for each lito value
variables|❎|select which variables from hard data will be interpolated in the soft data
variogram|☑️|a json or yaml file with variogram parameters. check notes for details.
output|☑️|save updated grid on this file path
display||show results in a chart window
## 📓 Notes
 - Empty grids can be created using the tool vtk_create_grid.py
 - Hard data *must* contain x,y,z columns, either with those exact names or popular synonyms. Ex.: x, easting, mid_x, leste
 - Results can be visualized with tools such as db_voxel_view.py or softwares such as Paraview and F3D.

### Possible Variogram parameters and their default values
name|default
---|---
algorithm|ordinary
variogram_model|gaussian
variogram_parameters|None
nlags|6
anisotropy_scaling_y|1.0
anisotropy_scaling_z|1.0
anisotropy_angle_x|0.0
anisotropy_angle_y|0.0
anisotropy_angle_z|0.0

## 📚 Examples
### input hard data
![screenshot3](https://github.com/pemn/assets/blob/main/vtk_krig3.png?raw=true)
### output estimade grid
![screenshot2](https://github.com/pemn/assets/blob/main/vtk_krig2.png?raw=true)
## 🧩 Compatibility
distribution|status
---|---
![winpython_icon](https://github.com/pemn/assets/blob/main/winpython_icon.png?raw=true)|✔
![vulcan_icon](https://github.com/pemn/assets/blob/main/vulcan_icon.png?raw=true)|❌
![anaconda_icon](https://github.com/pemn/assets/blob/main/anaconda_icon.png?raw=true)|✔
## 🙋 Support
Any question or problem contact:
 - paulo.ernesto
## 💎 License
Apache 2.0
Copyright ![vale_logo_only](https://github.com/pemn/assets/blob/main/vale_logo_only_r.svg?raw=true) Vale 2023
