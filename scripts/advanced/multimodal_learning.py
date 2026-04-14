# Apply Rodrigues' rotation formula
            coords[k] = coords[k] + np.array([
                curvature * axis[1],
                -curvature * axis[0],