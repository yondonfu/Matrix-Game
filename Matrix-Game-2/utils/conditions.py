
import torch
import random
import math

def combine_data(data, num_frames=57, keyboard_dim=6, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse == True:
        mouse_condition = torch.zeros((num_frames, 2))
    
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]['keyboard_condition']
        if mouse == True:
            m = data[rd]['mouse_condition']
        
        if current_frame == 0:
            keyboard_condition[:1] = k[:1]
            if mouse == True:
                mouse_condition[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame+rd_frame] = k.repeat(repeat_time, 1)
            if mouse == True:
                mouse_condition[current_frame:current_frame+rd_frame] = m.repeat(repeat_time, 1)
            current_frame += rd_frame
    if mouse == True:
        return {
                "keyboard_condition": keyboard_condition,
                "mouse_condition": mouse_condition
            }
    return {"keyboard_condition": keyboard_condition}

def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        # "back",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        # "back_left",
        # "back_right",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
        # "camera_ur",
        # "camera_ul",
        # "camera_dl",
        # "camera_dr" 
        # "camera_up",
        # "camera_down",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in (actions_single_action + actions_double_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)


def Bench_actions_gta_drive(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "back",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in (actions_single_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
    }
    
    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)

def Bench_actions_templerun(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "jump",
        "slide",
        "leftside",
        "rightside",
        "turnleft",
        "turnright",
        "nomove"
    ]

    actions_to_test = actions_single_action

    base_action = actions_single_action

    KEYBOARD_IDX = { 
        "nomove": 0, "jump": 1, "slide": 2, "turnleft": 3,
        "turnright": 4, "leftside": 5, "rightside": 6
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)

def Bench_orbit_fixed_point(num_frames, orbit_radius=2.0, orbit_speed=1.0, num_samples_per_action=4):
    """
    Generate action sequence for orbiting around a fixed point while keeping camera focused on it.
    
    Args:
        num_frames: Total number of frames to generate
        orbit_radius: Radius of the circular orbit (affects movement intensity)
        orbit_speed: Speed of orbit (1.0 = one full circle over all frames)
        num_samples_per_action: Samples per action frame (for consistency with other Bench_ functions)
    """
    
    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }
    
    CAM_VALUE = 0.1
    
    # Calculate the number of action frames we need
    num_action_frames = (num_frames - 1) // 4 + 1
    
    data = []
    
    for frame_idx in range(num_action_frames):
        # Calculate current angle in the orbit
        angle = (frame_idx / max(1, num_action_frames - 1)) * 2 * math.pi * orbit_speed
        
        # Calculate position on circle (x, z plane, y is up which we don't change)
        x = orbit_radius * math.cos(angle)
        z = orbit_radius * math.sin(angle)
        
        # Calculate movement direction based on tangent to circle
        # Tangent vector gives us the direction we should move
        tangent_x = -orbit_radius * math.sin(angle)
        tangent_z = orbit_radius * math.cos(angle)
        
        # Normalize tangent vector
        tangent_length = math.sqrt(tangent_x**2 + tangent_z**2)
        if tangent_length > 0:
            tangent_x /= tangent_length
            tangent_z /= tangent_length
        
        # Convert tangent to movement commands
        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)]
        
        # Forward/back component (z-axis)
        if tangent_z > 0.3:  # Moving forward
            for row in keyboard_condition:
                row[KEYBOARD_IDX["forward"]] = 1
        elif tangent_z < -0.3:  # Moving backward
            for row in keyboard_condition:
                row[KEYBOARD_IDX["back"]] = 1
        
        # Left/right component (x-axis)
        if tangent_x > 0.3:  # Moving right
            for row in keyboard_condition:
                row[KEYBOARD_IDX["right"]] = 1
        elif tangent_x < -0.3:  # Moving left
            for row in keyboard_condition:
                row[KEYBOARD_IDX["left"]] = 1
        
        # Handle diagonal movement - combine forward/back with left/right
        if abs(tangent_x) > 0.1 and abs(tangent_z) > 0.1:
            # For diagonal movement, use both keys with reduced threshold
            if tangent_z > 0.1:  # Forward component
                for row in keyboard_condition:
                    row[KEYBOARD_IDX["forward"]] = 1
            elif tangent_z < -0.1:  # Back component
                for row in keyboard_condition:
                    row[KEYBOARD_IDX["back"]] = 1
                    
            if tangent_x > 0.1:  # Right component
                for row in keyboard_condition:
                    row[KEYBOARD_IDX["right"]] = 1
            elif tangent_x < -0.1:  # Left component
                for row in keyboard_condition:
                    row[KEYBOARD_IDX["left"]] = 1
        
        # Calculate camera rotation to keep looking at center (0,0,0)
        # We need to look from our position (x, z) toward the center (0, 0)
        look_angle = math.atan2(-x, -z)  # Angle to look toward center
        camera_angle = math.atan2(-tangent_x, -tangent_z)  # Current movement direction
        
        # Calculate the angle difference we need to rotate the camera
        angle_diff = look_angle - camera_angle
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Convert angle difference to mouse movement
        # Positive angle_diff means we need to turn right, negative means left
        mouse_horizontal = angle_diff * CAM_VALUE * 2  # Scale the rotation
        
        # Clamp mouse movement to reasonable values
        mouse_horizontal = max(-CAM_VALUE * 2, min(CAM_VALUE * 2, mouse_horizontal))
        
        # Create mouse condition (vertical=0 as requested, horizontal=calculated)
        mouse_condition = [[0, mouse_horizontal] for _ in range(num_samples_per_action)]
        
        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)