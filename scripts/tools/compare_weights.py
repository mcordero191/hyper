import os, re, glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def load_weights_flattened_custom(h5_path):
    """Loads all weights from a custom Keras HDF5 file and flattens into a 1D array."""
    weights = []
    with h5py.File(h5_path, 'r') as f:
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights.append(np.array(obj).flatten())
        f.visititems(visit_fn)
    return np.concatenate(weights)

def compute_weight_differences(weight_paths):
    """Compute L2 and cosine distances between consecutive weights."""
    flattened_weights = [load_weights_flattened_custom(p) for p in weight_paths]
    
    l2_diffs = [np.linalg.norm(flattened_weights[i+1] - flattened_weights[i])
                for i in range(len(flattened_weights) - 1)]
    cosine_diffs = [cosine(flattened_weights[i+1], flattened_weights[i])
                    for i in range(len(flattened_weights) - 1)]
    
    return l2_diffs, cosine_diffs

def visualize_weight_differences(weight_dir, pattern="h20230323_090000_i000_v1.3.1.h5.{:05d}.weights.h5", steps=range(500, 10000, 500)):
    
    """Load weights, compute differences, and plot the metrics."""
    paths = [os.path.join(weight_dir, pattern.format(step-1)) for step in steps]
    l2_diffs, cosine_diffs = compute_weight_differences(paths)
    
    steps_trimmed = steps[1:]

    plt.figure()
    plt.plot(steps_trimmed, l2_diffs, label='L2 Norm Difference')
    plt.plot(steps_trimmed, cosine_diffs, label='Cosine Distance')
    plt.xlabel('Training Step')
    plt.title('Weight Changes Between Snapshots')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_weights_per_layer(h5_path):
    """Loads all weights from a custom Keras HDF5 file, returns a dictionary {layer_name: flattened_weights}."""
    layer_weights = {}
    with h5py.File(h5_path, 'r') as f:
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Filter out optimizer weights or other non-layer weights if needed
                if "optimizer_weights" not in name and "optimizer" not in name:
                    # Use top-level group name as layer name
                    name_list = name.split('/')[0:3]
                    weight_data = np.array(obj).flatten()
                    
                    layer_name = "+".join(name_list) #name #"%s-%s-%s" %(root_layer, sub_layer1, sub_layer2)
                    
                    if layer_name in layer_weights:
                        layer_weights[layer_name].append(weight_data)
                    else:
                        layer_weights[layer_name] = [weight_data]
        f.visititems(visit_fn)
    
    # Concatenate weight parts for each layer
    for layer in layer_weights:
        layer_weights[layer] = np.concatenate(layer_weights[layer])
    
    return layer_weights

def compute_layerwise_differences(weight_paths):
    """Compute L2 norm differences per layer across checkpoints."""
    layerwise_diffs = {}
    layerwise_amp = {}

    # Load all snapshots
    all_weights = [load_weights_per_layer(path) for path in weight_paths]

    # Get union of all layer names
    all_layer_names = set()
    for weights in all_weights:
        all_layer_names.update(weights.keys())

    for layer in sorted(all_layer_names):
        diffs = []
        amp = []
        for i in range(len(all_weights) - 1):
            w1 = all_weights[i].get(layer)
            w2 = all_weights[i+1].get(layer)
            if w1 is not None and w2 is not None:
                diff = np.linalg.norm(w2 - w1)
                diffs.append(diff)
            else:
                diffs.append(np.nan)  # handle missing layer
            amp.append(np.linalg.norm(w1))
                
        layerwise_diffs[layer] = diffs
        layerwise_amp[layer] = amp

    return layerwise_diffs, layerwise_amp

def visualize_layerwise_changes(weight_dir, pattern="h*.*.weights.h5", steps=range(500, 4001, 500)):
    """Visualize per-layer weight differences."""
    
    paths = sorted( glob.glob(os.path.join(weight_dir, pattern)) )
    
    # paths = [os.path.join(weight_dir, pattern.format(step-1)) for step in steps]
    layer_diffs, layerwise_amp = compute_layerwise_differences(paths)
    steps_trimmed = [re.sub(r"\D", "",path)[-6:-1] for path in paths][1:]#steps[1:]

    a_max = np.max(layerwise_amp)
    
    for i, layer in enumerate( layer_diffs.keys() ):
        
        diffs = layer_diffs[layer]
        amp = layerwise_amp[layer]
        
        plt.subplot(211)
        plt.plot(steps_trimmed, diffs, marker='o')
        plt.title(f'Layer: {layer}')
        plt.xlabel('Training Step')
        plt.ylabel('L2 Norm Difference')
        plt.grid(True)
        
        plt.subplot(212)
        plt.plot(steps_trimmed, amp, marker='x')
        plt.title(f'Layer: {layer}')
        plt.xlabel('Training Step')
        plt.ylabel('L2 Norm Amplitude')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
def load_weights_per_layer2(h5_path):
    layer_weights = {}
    with h5py.File(h5_path, 'r') as f:
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Dataset):
                if "optimizer" in name: 
                    return
                
                # root = name.split('/')[0]
                
                name_list = name.split('/')[0:2]
                layer_name = "+".join(name_list)
                
                arr = np.array(obj).ravel()
                layer_weights.setdefault(layer_name, []).append(arr)
                
        f.visititems(visit_fn)
    # flatten each layer
    return {L: np.concatenate(parts) for L,parts in layer_weights.items()}

def compute_layerwise_differences2(weight_paths):
    all_weights = [load_weights_per_layer2(p) for p in weight_paths]
    layers = sorted({ L for w in all_weights for L in w.keys() })
    diffs = {L: [] for L in layers}
    for i in range(len(all_weights)-1):
        w1,w2 = all_weights[i], all_weights[i+1]
        for L in layers:
            v1,v2 = w1.get(L), w2.get(L)
            diffs[L].append(np.nan if v1 is None or v2 is None
                             else np.linalg.norm(v2-v1))
    return diffs

def compute_layerwise_norm(weight_paths):
    
    all_weights = [load_weights_per_layer2(p) for p in weight_paths]
    
    layers = sorted({ L for w in all_weights for L in w.keys() })
    
    norm = {L: [] for L in layers}
    
    for i in range(len(all_weights)):
        w1 = all_weights[i]
        for L in layers:
            v1 = w1.get(L)
            norm[L].append(np.linalg.norm(v1))
    return norm

def plot_scale_contributions(weight_dir,
                             pattern="h*.*.weights.h5",
                             nscales=5):
    # 1) gather checkpoints
    paths = sorted(glob.glob(os.path.join(weight_dir,pattern)))
    steps = [int(re.sub(r"\D", "",path)[-6:-1]) for path in paths][:]
    # 2) per-layer diffs
    layer_diffs = compute_layerwise_norm(paths)
    # 3) assign each layer to a scale: look for "scale_net_i" in the layer key
    scale_diffs = {i: [] for i in range(nscales)}
    for layer_name, diffs in layer_diffs.items():
        m = re.search(r"scale_net", layer_name)
        if m:
            si = layer_name[m.span()[1]+1:m.span()[1]+2]
            try: si = int(si)
            except: si = 0
            
            if 0 <= si < nscales:
                scale_diffs[si].append(diffs)
    # 4) aggregate per scale (mean across layers)
    agg = {}
    for si, mats in scale_diffs.items():
        if not mats:
            agg[si] = np.zeros(len(steps))
        else:
            agg[si] = np.nanmean(np.vstack(mats), axis=0)
    # 5) plot
    plt.figure(figsize=(8,5))
    for si, vals in agg.items():
        plt.plot(steps[:], vals, '-o', label=f"scale {si}")
    plt.xlabel("Training step")
    plt.ylabel("mean ΔL₂ per-layer")
    plt.title("Per-scale weight‐update magnitudes")
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    
    path = "/Users/radar/Data/IAP/SIMONe/Virtual/ICON_20160815/ICON_+00+70+90/hMULT_VP_divl04.01.064_w1.0e-05lr1.0e-03lf0ur1.0e-06T24transf/c20160815/log"
    # visualize_layerwise_changes(path)
    
    plot_scale_contributions(path)