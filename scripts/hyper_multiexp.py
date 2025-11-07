from multiprocessing import Process
import subprocess
import sys

def run_experiment(args_list):
    cmd = [sys.executable, "1.0.hyperMLT.py"] + args_list
    subprocess.run(cmd)

if __name__ == "__main__":
    experiments = [
        # ["--exp", "vortex", "--pde-upd-rate", "1e-7", "--pde-ratio", "1e+1", "--dropout", "0"],
        # ["--exp", "vortex", "--pde-upd-rate", "1e-7", "--pde-ratio", "1e+1", "--dropout", "0", "-neurons-per_layer", "256"],
        # ["--exp", "vortex", "--pde-upd-rate", "1e-8", "--pde-ratio", "1e+1", "--dropout", "0", "--neurons-per_layer", "256"],
        # ["--exp", "vortex", "--pde-upd-rate", "1e-9", "--pde-ratio", "1e+2", "--dropout", "0", "--neurons-per_layer", "256"],
        ["--exp", "ext24", "--pde-upd-rate", "5e-10", "--pde-ratio", "1e+2", "--dropout", "0", "--neurons-per_layer", "256"],
        ["--exp", "ext24", "--pde-upd-rate", "1e-10", "--pde-ratio", "1e+2", "--dropout", "0", "--neurons-per_layer", "256"],
        ["--exp", "ext24", "--pde-upd-rate", "1e-9", "--pde-ratio", "1e+2", "--dropout", "0", "--neurons-per_layer", "256"],
        # ["--exp", "ext24", "--pde-upd-rate", "1e-8", "--pde-ratio", "1e+1", "--dropout", "0"],
    ]
    
    procs = []
    for args in experiments:
        p = Process(target=run_experiment, args=(args,))
        p.start()
        p.join()
        
    #     procs.append(p)
    #
    # for p in procs:
    #     p.join()