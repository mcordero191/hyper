# hyper
A PINN approach to reconstruct 3D wind fields from meteor measurements

# How to test the code
- Download the HYPER code and meteor data file
- Execute:
  
  $ scripts/hyperMLT.py --dpath=[location of meteor data files] --rpath=[folder where HYPER's weights will be stored]
  
  * HYPER training will begin
  * A keras file will be saved in "rpath"
- Run the following python script to visualize wind estimates at a given LAT/LON location. If not lat/lon are provided, the center of the domain will be selected.

  $ script/plot_hyper_winds.py --path=[folder where HYPER's weights were stored] --lat=[LATITUDE] --lon=[LONGITUDE]
