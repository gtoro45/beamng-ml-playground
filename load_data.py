import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path, val_split=0.1, test_split=0.2, input_dims=None, use_speed_as_input=False):
    '''
    Loads custom dataset dumped in CIFAR format from a single file and partitions
    it into train, validation, and test sets. This functions, conceptually, 
    returns data in {image : label} or {image, speed : label} pairs, depending on 
    the use_speed_as_input flag
    
    Args:
        file_path: String or Path. Path to the single pickle (.pkl) file.
        val_split: Float. Fraction of the total dataset for validation (e.g., 0.1 for 10%).
        test_split: Float. Fraction of the total dataset for testing (e.g., 0.2 for 20%).
        input_dims: Tuple or List of 2 ints: (width, height). Fallback image
                    dimensions used if metadata is absent.
        use_speed_as_input: Boolean. If True, returns input speed along with images
                            and excludes speed from targets. If False, returns image 
                            inputs only.

    Returns:
        If use_speed_as_input is False:
            x_train, y_train: Arrays of shape [N_train, 3, H, W] and [N_train, 3].
            x_val, y_val:     Arrays of shape [N_val, 3, H, W] and [N_val, 3].
            x_test, y_test:   Arrays of shape [N_test, 3, H, W] and [N_test, 3].
            
        If use_speed_as_input is True:
            (x_train_img, x_train_speed), y_train: 
                x_train_img: [N_train, 3, H, W], 
                x_train_speed: [N_train, 1], 
                y_train: [N_train, 3]
            (x_val_img, x_val_speed), y_val:     
                x_val_img: [N_val, 3, H, W], 
                x_val_speed: [N_val, 1], 
                y_val: [N_val, 3]
            (x_test_img, x_test_speed), y_test:   
                x_test_img: [N_test, 3, H, W], 
                x_test_speed: [N_test, 1], 
                y_test: [N_test, 3]
    '''
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must sum to less than 1.0")

    with open(file_path, 'rb') as fo: 
        batch = pickle.load(fo, encoding='bytes')
        
    x_all = batch[b'data']                                          # Shape: (N, 3 * H * W)
    telemetry_all = np.array(batch[b'labels'], dtype=np.float32)    # Shape: (N, 4)
    w, h = batch.get(b'dims', input_dims)
    
    # Reshape images to planar format (N, 3, H, W)
    x_img_all = np.reshape(x_all, (-1, 3, h, w)).astype(np.uint8)
    
    # ALWAYS exclude speed from the output target array 'y'
    y_all = telemetry_all[:, :3]                                    # Shape: (N, 3) [steer, throttle, brake]
    speed_all = telemetry_all[:, 3:]                                # Shape: (N, 1) [speed]
    
    # Sequential split boundaries
    num_samples = x_img_all.shape[0]
    train_end = int(num_samples * (1.0 - val_split - test_split))
    val_end = int(num_samples * (1.0 - test_split))
    
    # Slice partitions
    x_tr_img, x_val_img, x_te_img = x_img_all[:train_end], x_img_all[train_end:val_end], x_img_all[val_end:]
    y_tr, y_val, y_te             = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]
    
    if use_speed_as_input:
        s_tr, s_val, s_te = speed_all[:train_end], speed_all[train_end:val_end], speed_all[val_end:]
        return (x_tr_img, s_tr), y_tr, (x_val_img, s_val), y_val, (x_te_img, s_te), y_te
    else:
        # Standard LeNet mode: returns raw image arrays and sliced targets
        return x_tr_img, y_tr, x_val_img, y_val, x_te_img, y_te


def view_data(file_path, input_dims=None):
    '''
    Opens an interactive Matplotlib window to scroll through images and telemetry
    stored in the dataset pickle file.
    
    Controls:
        - Right Arrow / Down Arrow / Scroll Down: Next image
        - Left Arrow / Up Arrow / Scroll Up: Previous image
        - 'q': Close window
    '''
    with open(file_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    x_raw = batch[b'data']
    y_raw = batch[b'labels']
    w, h = batch.get(b'dims', input_dims)

    # Reshape from planar (N, 3, H, W) to display format (N, H, W, 3)
    images = np.reshape(x_raw, (-1, 3, h, w)).transpose(0, 2, 3, 1)
    labels = np.array(y_raw, dtype=np.float32)
    total_frames = len(images)

    if total_frames == 0:
        print("Dataset is empty.")
        return

    current_idx = [0]  # List wrapper for mutable state in event handlers

    fig, ax = plt.subplots(figsize=(6, 6))
    img_display = ax.imshow(images[0])
    ax.axis('off')

    def update_frame(idx):
        idx = max(0, min(idx, total_frames - 1))
        current_idx[0] = idx
        img_display.set_data(images[idx])
        
        steer, throttle, brake, speed = labels[idx]
        ax.set_title(
            f"Frame: {idx + 1}/{total_frames}\n"
            f"Steering: {steer:+.2f} | Throttle: {throttle:.2f} | "
            f"Brake: {brake:.2f} | Speed: {speed:.1f}",
            fontsize=10
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ('right', 'down'):
            update_frame(current_idx[0] + 1)
        elif event.key in ('left', 'up'):
            update_frame(current_idx[0] - 1)
        elif event.key == 'q':
            plt.close(fig)

    def on_scroll(event):
        if event.button == 'down':
            update_frame(current_idx[0] + 1)
        elif event.button == 'up':
            update_frame(current_idx[0] - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    update_frame(0)
    plt.tight_layout()
    plt.show()