import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.draw import disk

# Parameters
default_num_samples = 10000
default_seq_len = 10
default_img_size = 28
default_train_ratio = 0.8
default_out_dir = Path(__file__).parent.parent.parent / 'data' / 'sprites'
default_out_dir.mkdir(parents=True, exist_ok=True)

# Size trajectory functions
def size_trajectory(t, phase, traj_type):
    # t: array of timesteps in [0, 1]
    # phase: float in [0, 1]
    # traj_type: int (0: up, 1: down, 2: V, 3: Lambda, 4: sinusoidal)
    if traj_type == 0:  # always increase
        return 0.3 + 0.5 * t
    elif traj_type == 1:  # always decrease
        return 0.8 - 0.5 * t
    elif traj_type == 2:  # V shape (decrease then increase)
        return 0.8 - 0.5 * np.abs(2 * t - 1)
    elif traj_type == 3:  # Lambda shape (increase then decrease)
        return 0.3 + 0.5 * np.abs(2 * t - 1)
    elif traj_type == 4:  # sinusoidal
        return 0.55 + 0.25 * np.sin(2 * np.pi * (t + phase))
    else:
        raise ValueError('Unknown trajectory type')

# HSV to RGB
from matplotlib.colors import hsv_to_rgb

def draw_circle(img, center, radius, color):
    rr, cc = disk(center, radius, shape=img.shape[:2])
    img[rr, cc, :] = color
    return img

def generate_colored_circles_dataset(num_samples=default_num_samples, seq_len=default_seq_len, img_size=default_img_size, out_dir=default_out_dir, train_ratio=default_train_ratio):
    np.random.seed(42)
    hues = np.random.uniform(0, 1, size=num_samples)
    phases = np.random.uniform(0, 1, size=num_samples)
    traj_types = np.random.choice([0, 1, 2, 3, 4], size=num_samples)
    # 0: up, 1: down, 2: V, 3: Lambda, 4: sinusoidal

    data = np.zeros((num_samples, seq_len, img_size, img_size, 3), dtype=np.float32)
    params = []
    t_arr = np.linspace(0, 1, seq_len)
    center = (img_size // 2, img_size // 2)
    max_radius = img_size // 2 - 2

    for i in tqdm(range(num_samples), desc='Generating samples'):
        hue = hues[i]
        phase = phases[i]
        traj_type = traj_types[i]
        color = hsv_to_rgb([hue, 1, 1])
        radii = size_trajectory(t_arr, phase, traj_type) * max_radius
        for t in range(seq_len):
            img = np.zeros((img_size, img_size, 3), dtype=np.float32)
            img = draw_circle(img, center, radii[t], color)
            data[i, t] = img
        params.append({'hue': float(hue), 'phase': float(phase), 'traj_type': int(traj_type)})

    # Normalize to [0,1]
    data = np.clip(data, 0, 1)
    # Convert to torch tensor, [N, T, C, H, W]
    data_tensor = torch.from_numpy(data).permute(0, 1, 4, 2, 3)  # [N, T, 3, H, W]

    # Split train/test
    n_train = int(num_samples * train_ratio)
    train_data = data_tensor[:n_train]
    test_data = data_tensor[n_train:]
    train_params = params[:n_train]
    test_params = params[n_train:]

    # Save
    torch.save(train_data, out_dir / 'ColoredCircles_train.pt')
    torch.save(test_data, out_dir / 'ColoredCircles_test.pt')
    torch.save(train_params, out_dir / 'ColoredCircles_train_params.pt')
    torch.save(test_params, out_dir / 'ColoredCircles_test_params.pt')
    print(f"✅ Saved train and test datasets to {out_dir}")

    # Visualization
    create_sample_visualization(train_data, train_params, out_dir / 'ColoredCircles_samples.png')

def create_sample_visualization(data_tensor, params, out_path, n_samples=5):
    # data_tensor: [N, T, 3, H, W]
    n_samples = min(n_samples, data_tensor.shape[0])
    seq_len = data_tensor.shape[1]
    fig, axes = plt.subplots(n_samples, seq_len, figsize=(seq_len * 2, n_samples * 2))
    for i in range(n_samples):
        for t in range(seq_len):
            img = data_tensor[i, t].permute(1, 2, 0).cpu().numpy()
            axes[i, t].imshow(img)
            axes[i, t].axis('off')
            if t == 0:
                p = params[i]
                axes[i, t].set_title(f"hue={p['hue']:.2f}\nphase={p['phase']:.2f}\ntype={p['traj_type']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved sample visualization to {out_path}")

if __name__ == '__main__':
    generate_colored_circles_dataset() 