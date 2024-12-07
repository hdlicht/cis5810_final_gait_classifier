import torch

def calculate_angles(marks):
    # Unflatten marks from Nx69 to Nx23x3
    marks = marks.view(-1, 23, 3)
    
    # Define the key points
    key_points = {
        'nose': 0,
        'left_shoulder': 1,
        'right_shoulder': 2,
        'left_elbow': 3,
        'right_elbow': 4,
        'left_wrist': 5,
        'right_wrist': 6,
        'left_hip': 13,
        'right_hip': 14,
        'left_knee': 15,
        'right_knee': 16,
        'left_ankle': 17,
        'right_ankle': 18,
        'left_toe': 21,
        'right_toe': 22
    }
    
    # Joints for which to calculate angles
    joints = [
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_hip', 'left_shoulder', 'left_elbow'),
        ('right_hip', 'right_shoulder', 'right_elbow'),
        ('left_shoulder', 'left_hip', 'left_knee'),
        ('right_shoulder', 'right_hip', 'right_knee'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_knee', 'left_ankle', 'left_toe'),
        ('right_knee', 'right_ankle', 'right_toe')
    ]
    
    # Collect indices for all points in the joints
    indices = torch.tensor([[key_points[a], key_points[b], key_points[c]] for a, b, c in joints], dtype=torch.long)
    
    # Extract the relevant points
    a_points = marks[:, indices[:, 0]]  # NxJx3
    b_points = marks[:, indices[:, 1]]  # NxJx3
    c_points = marks[:, indices[:, 2]]  # NxJx3
    
    # Calculate vectors
    ba_vectors = a_points - b_points  # NxJx3
    bc_vectors = c_points - b_points  # NxJx3
    
    # Normalize vectors
    ba_norm = ba_vectors.norm(dim=2, keepdim=True)  # NxJx1
    bc_norm = bc_vectors.norm(dim=2, keepdim=True)  # NxJx1
    
    ba_normalized = ba_vectors / ba_norm  # NxJx3
    bc_normalized = bc_vectors / bc_norm  # NxJx3
    
    # Compute cosine of the angle
    cos_angles = torch.einsum('ijk,ijk->ij', ba_normalized, bc_normalized)  # NxJ
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)  # Ensure numerical stability
    
    # Compute angles in degrees
    angles = torch.acos(cos_angles) 
    
    return angles

# Test the function
data = torch.load('data/gavd_dataset/all_features/cljarhldg00d13n6l7utw0lqn_frames_210_519.pt')
marks = data[:, 1:70]  # Example tensor (100 samples, 69 values)
angles = calculate_angles(marks)
print(angles.shape)  # Expected shape: (100, 12), one angle per joint for each sample

# plot the x-th angle over time
import matplotlib.pyplot as plt
x = 2
plt.plot(angles[:, x])
plt.xlabel('Sample')
plt.ylabel('Angle (degrees)')
plt.title('Angle over time')
plt.show()