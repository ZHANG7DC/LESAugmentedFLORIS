import os
import yaml
import numpy as np
from floris.tools import FlorisInterface
import argparse

def load_config(config_path):
    """Load configuration from the config.yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_3d_flowfield(fi, resolution, H_grid, wind_speed, x, y, yaw_angles, bounds, output_path):
    """Generate a 3D flow field and save it to the specified path."""
    fi.reinitialize(
        layout_x=x,
        layout_y=y,
        wind_directions=[270],
        wind_speeds=[wind_speed]
    )
    fi.calculate_wake()

    # Generate slices at specified heights
    slices = []
    for height in H_grid:
        horizontal_plane = fi.calculate_horizontal_plane(
            wd=[270.0], height=height, 
            x_bounds=(bounds[0],bounds[1]), y_bounds=(bounds[2],bounds[3]),
            x_resolution=resolution[2], y_resolution=resolution[1], yaw_angles=yaw_angles
        )
        slices.append(horizontal_plane.df.u.values.reshape(resolution[1], resolution[2]))

    # Stack slices into a 3D array and save as .npy
    UF = np.stack(slices)
    np.save(output_path, np.nan_to_num(UF, nan=0.0))
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate 3D flow fields from config and GCH files.')
    parser.add_argument('--config', required=True, help='Path to the config.yaml file.')
    args = parser.parse_args()

    # Load configuration from config.yaml
    config = load_config(args.config)

    # Load H_grid from the specified .npy file
    H_grid = np.load(config['H_grid_path'])

    # Initialize the FlorisInterface with gch.yaml
    fi = FlorisInterface(config['gch_path'])

    # Create output directory if it doesn't exist
    output_dir = config['save_path']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate flow fields
    create_3d_flowfield(
                fi, config['resolution'], H_grid, config['wind_speed'], config['yaw_angple'], config['bounds'], config['save_path']
            )

if __name__ == '__main__':
    main()
