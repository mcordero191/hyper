# hyper
A PINN approach to reconstruct 3D wind fields from meteor measurements

# How to test the code
- Download the meteor data file
- Execute:
  $ scripts/hyperMLT.py --dpath=<location of the meteor data file> --rpath=<path were the trained neural network will be stored>
  
  * Hyper training will begin
  * A keras file will be saved in "rpath"
- Execute the following python script to visualize estimated winds in time vs altitude at a given LAT/LON location. If not lat/lon are selected, the selected point will be located at the center of the domain.
  $ script/plot_hyper_winds.py --path=<path were the trained neural network was stored> --lat=<LATITUDE> --lon=<LONGITUDE>
