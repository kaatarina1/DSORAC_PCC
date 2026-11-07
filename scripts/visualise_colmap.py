import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    q = np.array(q)
    q = q / np.linalg.norm(q)  # Normalize
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def parse_images_file(images_path):
    """Parse COLMAP images.txt file"""
    cameras = []
    
    with open(images_path, 'r') as f:
        lines = f.readlines()
        
        # The file has a specific structure: every other line is image data
        for i in range(0, len(lines)):
            if lines[i].strip().startswith('#'):
                continue
                
            parts = lines[i].split(' ')
            if len(parts) < 10:
                continue
                
            # Extract parameters
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = ' '.join(parts[9:])
            
            # Calculate camera center (world coordinates)
            R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            camera_center = -R.T @ t
            
            cameras.append({
                'image_id': image_id,
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': camera_id,
                'name': image_name,
                'R': R,
                'center': camera_center
            })
    
    return cameras

def create_camera_mesh(center, R, scale=0.1):
    """Create a camera mesh for visualization"""
    # Define camera points in camera coordinate system
    points = np.array([
        [0, 0, 0],          # Center
        [scale, scale, scale*2],   # Top-right
        [scale, -scale, scale*2],  # Bottom-right
        [-scale, -scale, scale*2], # Bottom-left
        [-scale, scale, scale*2]   # Top-left
    ])
    
    # Transform points to world coordinate system
    points = (R @ points.T).T + center
    
    # Define edges of the camera pyramid
    vertices = [
        [points[0], points[1], points[2]],
        [points[0], points[2], points[3]],
        [points[0], points[3], points[4]],
        [points[0], points[4], points[1]],
        [points[1], points[2], points[3], points[4]]
    ]
    
    return vertices

def visualize_cameras(cameras, output_path=None):
    """Visualize camera positions and orientations"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each camera
    colors = plt.cm.tab10(np.linspace(0, 1, len(cameras)))
    
    for i, camera in enumerate(cameras):
        center = camera['center']
        R = camera['R']
        
        # Create and plot camera mesh
        camera_mesh = create_camera_mesh(center, R)
        mesh = Poly3DCollection(camera_mesh, alpha=0.3, linewidths=1, edgecolors='k')
        mesh.set_facecolor(colors[i % len(colors)])
        ax.add_collection3d(mesh)
        
        # Plot camera center
        ax.scatter(center[0], center[1], center[2], c='r', marker='o', s=30)
        
        # Add camera name as label
        ax.text(center[0], center[1], center[2], camera['name'], fontsize=8)
        
        # Draw camera orientation vectors
        vec_scale = 0.2
        x_dir = R[:, 0] * vec_scale
        y_dir = R[:, 1] * vec_scale
        z_dir = R[:, 2] * vec_scale * 2  # Make forward direction longer
        
        ax.quiver(center[0], center[1], center[2], 
                  x_dir[0], x_dir[1], x_dir[2], color='r', length=vec_scale)
        ax.quiver(center[0], center[1], center[2], 
                  y_dir[0], y_dir[1], y_dir[2], color='g', length=vec_scale)
        ax.quiver(center[0], center[1], center[2], 
                  z_dir[0], z_dir[1], z_dir[2], color='b', length=vec_scale*2)
    
    # Set labels and title
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Forward)')
    ax.set_title('COLMAP Camera Poses Visualization')
    
    # Set equal aspect ratio
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    
    # Add legend for axis colors
    ax.legend([plt.Line2D([0], [0], color='r', lw=2),
               plt.Line2D([0], [0], color='g', lw=2),
               plt.Line2D([0], [0], color='b', lw=2)], 
              ['X-axis (Right)', 'Y-axis (Down)', 'Z-axis (Forward)'])
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot the cameras of a COLMAP images.txt")
    parser.add_argument("images_txt", nargs="?", default=os.path.join(os.path.dirname(__file__), 'images.txt'),
                        help="e.g. data/<scene>/sparse/0/images.txt (default: images.txt next to this script)")
    images_path = parser.parse_args().images_txt
    
    print("Parsing COLMAP images.txt file...")
    cameras = parse_images_file(images_path)
    
    print(f"Found {len(cameras)} cameras")
    for camera in cameras[:5]:  # Show first 5 cameras as sample
        print(f"Camera {camera['image_id']}: {camera['name']}")
        print(f"  Position: {camera['center']}")
        print(f"  Rotation Matrix:\n{camera['R']}")
    
    print("\nVisualizing cameras...")
    visualize_cameras(cameras)

if __name__ == "__main__":
    main()