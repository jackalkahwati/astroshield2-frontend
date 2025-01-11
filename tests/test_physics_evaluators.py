[Previous code remains unchanged until the launch trajectory generation part...]

    # Test launch trajectory with improved physics
    trajectory = np.zeros((24, 6))
    dt = 1.0  # Time step
    
    # Initial conditions
    alt = 0.0
    v_vert = 0.0
    v_horiz = 0.0
    target_alt = 200000  # Target altitude
    
    for i in range(24):
        progress = i / 23
        
        if progress < 0.2:  # Initial vertical ascent
            # Even more gradual acceleration
            alt += v_vert * dt
            v_vert += 8 * dt  # Constant acceleration of ~0.8g
            
            trajectory[i] = [evaluator.R + alt, 0, 0, 0, v_vert, 0]
            
        else:  # Main ascent with gravity turn
            # Smooth transition to orbital velocity
            remaining_steps = 23 - i
            if remaining_steps > 0:
                # Calculate required vertical velocity for remaining altitude gain
                alt_remaining = target_alt - alt
                v_vert_required = alt_remaining / (remaining_steps * dt)  # Average velocity needed
                v_vert = min(v_vert, v_vert_required)  # Don't exceed required velocity
            else:
                # Final step - maintain current velocities
                v_vert = max(0, v_vert * 0.9)  # Gradually reduce vertical velocity
            
            # Gradually pitch over
            pitch_angle = np.pi/2 - np.pi/3 * ((progress-0.2)/0.8)
            
            # Calculate orbital velocity at current altitude
            r = evaluator.R + alt
            v_orbital = np.sqrt(evaluator.G * evaluator.M / r)
            
            # Gradually increase horizontal velocity
            v_horiz += 4 * dt  # Gentler horizontal acceleration
            v_horiz = min(v_horiz, v_orbital)  # Don't exceed orbital velocity
            
            # Update altitude
            alt += v_vert * dt
            
            # Update trajectory
            trajectory[i] = [
                (evaluator.R + alt) * np.cos(pitch_angle),
                (evaluator.R + alt) * np.sin(pitch_angle),
                0,
                -v_horiz * np.sin(pitch_angle),
                v_vert * np.cos(pitch_angle),
                0
            ]
    
    # Run tests
    results = []
    
[Rest of the code remains unchanged...]
