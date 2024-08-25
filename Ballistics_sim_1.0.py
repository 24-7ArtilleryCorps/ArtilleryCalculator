import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)

# Function to convert degrees to mils
def degrees_to_mils(degrees):
    return degrees * 17.7777777778

# Corrected function to convert mils to degrees
def mils_to_degrees(mils):
    return mils * 0.05625

def simulate_trajectory(angle_deg, target_range, launch_height, target_height,
                        projectile_mass=23, projectile_air_drag=0.0043, projectile_velocity=212.5):
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Initial conditions
    vx0 = projectile_velocity * np.cos(angle_rad)  # Initial velocity in x direction (m/s)
    vy0 = projectile_velocity * np.sin(angle_rad)  # Initial velocity in y direction (m/s)
    x0 = 0  # Initial position in x (m)
    y0 = launch_height  # Initial height of the launcher (m)

    # Time step for numerical integration
    dt = 0.01  # Time step (s)
    t_max = 100  # Maximum time to simulate (s)

    # Initialize arrays to store results
    x = [x0]
    y = [y0]
    vx = [vx0]
    vy = [vy0]

    # Numerical integration using Runge-Kutta method
    for step in np.arange(0, t_max, dt):
        # Current values
        x_current = x[-1]
        y_current = y[-1]
        vx_current = vx[-1]
        vy_current = vy[-1]

        # Calculate speed
        v = np.sqrt(vx_current ** 2 + vy_current ** 2)

        if v == 0:
            # To avoid division by zero when speed is zero
            drag_force = 0
        else:
            # Drag force calculation (in Newtons)
            drag_force = projectile_air_drag * v ** 2

        # Acceleration due to drag (negative because it's opposite to velocity)
        ax = -drag_force * vx_current / (projectile_mass * v)
        ay = -g - (drag_force * vy_current / (projectile_mass * v))

        # Runge-Kutta method (4th order)
        kx1 = vx_current * dt
        ky1 = vy_current * dt
        kvx1 = ax * dt
        kvy1 = ay * dt

        kx2 = (vx_current + kvx1 / 2) * dt
        ky2 = (vy_current + kvy1 / 2) * dt
        kvx2 = ax * dt
        kvy2 = ay * dt

        kx3 = (vx_current + kvx2 / 2) * dt
        ky3 = (vy_current + kvy2 / 2) * dt
        kvx3 = ax * dt
        kvy3 = ay * dt

        kx4 = (vx_current + kvx3) * dt
        ky4 = (vy_current + kvy3) * dt
        kvx4 = ax * dt
        kvy4 = ay * dt

        x_next = x_current + (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        y_next = y_current + (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6
        vx_next = vx_current + (kvx1 + 2 * kvx2 + 2 * kvx3 + kvx4) / 6
        vy_next = vy_current + (kvy1 + 2 * kvy2 + 2 * kvy3 + kvy4) / 6

        # Check if projectile has passed the target range or hit the ground
        if x_next > target_range and y_next < target_height:
            break

        # Append next values
        x.append(x_next)
        y.append(y_next)
        vx.append(vx_next)
        vy.append(vy_next)

    # Calculate time of flight
    time_of_flight = len(x) * dt

    # Return the final distance traveled, trajectory data, and time of flight
    return x, y, x[-1], y[-1], time_of_flight

def objective_function(angle_deg, target_range, launch_height, target_height,
                       projectile_mass, projectile_air_drag, projectile_velocity):
    _, _, distance_traveled, final_height, _ = simulate_trajectory(
        angle_deg, target_range, launch_height, target_height,
        projectile_mass, projectile_air_drag, projectile_velocity)
    return np.abs(distance_traveled - target_range) + np.abs(final_height - target_height)

def main(launch_height, target_height, target_range,
         projectile_mass=23, projectile_air_drag=0.0043, projectile_velocity=212.5):
    # Optimize to find the launch angle below 45 degrees
    result_below = minimize_scalar(
        objective_function,
        args=(target_range, launch_height, target_height, projectile_mass, projectile_air_drag, projectile_velocity),
        bounds=(0, 45),
        method='bounded'
    )
    optimal_angle_deg_below = result_below.x
    optimal_angle_mils_below = degrees_to_mils(optimal_angle_deg_below)
    _, _, _, _, time_of_flight_below = simulate_trajectory(
        optimal_angle_deg_below, target_range, launch_height, target_height,
        projectile_mass, projectile_air_drag, projectile_velocity)

    # Optimize to find the launch angle above 45 degrees
    result_above = minimize_scalar(
        objective_function,
        args=(target_range, launch_height, target_height, projectile_mass, projectile_air_drag, projectile_velocity),
        bounds=(45, 90),
        method='bounded'
    )
    optimal_angle_deg_above = result_above.x
    optimal_angle_mils_above = degrees_to_mils(optimal_angle_deg_above)
    _, _, _, _, time_of_flight_above = simulate_trajectory(
        optimal_angle_deg_above, target_range, launch_height, target_height,
        projectile_mass, projectile_air_drag, projectile_velocity)

    indirect_mils_adjusted = optimal_angle_mils_above - 17

    # Returning the values to JavaScript
    return {
        "direct_solution_mils": optimal_angle_mils_below,
        "direct_time_of_flight": time_of_flight_below,
        "indirect_solution_mils": indirect_mils_adjusted,
        "indirect_time_of_flight": time_of_flight_above
    }

if __name__ == "__main__":
    # Example input for testing
    result = main(launch_height=10, target_height=0, target_range=2000)
    print(result)
