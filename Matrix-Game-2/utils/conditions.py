
import torch
import random

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

def Bench_actions_universal_static(num_frames, num_samples_per_action=4):
    """
    Generate static (no action) conditioning data for universal mode.
    Creates conditioning that reflects no mouse movement and no keyboard input.
    """
    # Create single static action representing no input
    actions_to_test = ["no_action"]
    
    data = []
    
    for action_name in actions_to_test:
        # No keyboard input (all zeros for WASD)
        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        # No mouse movement (all zeros for camera)
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)] 

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)

def Bench_actions_universal_mixed(num_frames, num_samples_per_action=4, no_action_ratio=0.2):
    """
    Generate mixed conditioning data for universal mode that includes sequences of no action inputs.
    Starts and ends with actual action inputs, with no-action sequences in the middle.
    """
    # Simple approach: generate full sequences and splice them together
    # Calculate rough frame distribution  
    start_end_frames = max(5, int(num_frames * (1.0 - no_action_ratio) / 2))
    
    # Ensure frame counts are valid for combine_data (num_frames % 4 == 1)
    if start_end_frames % 4 != 1:
        start_end_frames = ((start_end_frames - 1) // 4) * 4 + 1
    
    middle_frames = max(1, num_frames - 2 * start_end_frames)
    if middle_frames % 4 != 1:
        middle_frames = ((middle_frames - 1) // 4) * 4 + 1
    
    # Generate sections using existing functions
    start_data = Bench_actions_universal(start_end_frames, num_samples_per_action)
    middle_data = Bench_actions_universal_static(middle_frames, num_samples_per_action) 
    end_data = Bench_actions_universal(start_end_frames, num_samples_per_action)
    
    # Concatenate sections
    keyboard_condition = torch.cat([
        start_data["keyboard_condition"],
        middle_data["keyboard_condition"], 
        end_data["keyboard_condition"]
    ], dim=0)
    
    mouse_condition = torch.cat([
        start_data["mouse_condition"],
        middle_data["mouse_condition"],
        end_data["mouse_condition"] 
    ], dim=0)
    
    # Ensure exact length
    current_length = keyboard_condition.shape[0]
    if current_length > num_frames:
        keyboard_condition = keyboard_condition[:num_frames]
        mouse_condition = mouse_condition[:num_frames]
    elif current_length < num_frames:
        # Pad with zeros if too short
        padding_frames = num_frames - current_length
        keyboard_pad = torch.zeros((padding_frames, 4))
        mouse_pad = torch.zeros((padding_frames, 2))
        keyboard_condition = torch.cat([keyboard_condition, keyboard_pad], dim=0)
        mouse_condition = torch.cat([mouse_condition, mouse_pad], dim=0)
    
    return {
        "keyboard_condition": keyboard_condition,
        "mouse_condition": mouse_condition
    }


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

def Bench_actions_gta_drive_static(num_frames, num_samples_per_action=4):
    """
    Generate static (no action) conditioning data for GTA drive mode.
    Creates conditioning that reflects no keyboard input and no mouse movement.
    """
    # Create single static action representing no input
    actions_to_test = ["no_action"]
    
    data = []
    
    for action_name in actions_to_test:
        # No keyboard input (all zeros for forward/back)
        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)] 
        # No mouse movement (all zeros for camera)
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)] 

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

def Bench_actions_templerun_static(num_frames, num_samples_per_action=4):
    """
    Generate static (no action) conditioning data for templerun mode.
    Creates conditioning that reflects no keyboard input or actions.
    """
    # Create single static action representing no input
    actions_to_test = ["no_action"]
    
    data = []
    
    for action_name in actions_to_test:
        # No keyboard input (all zeros for all 7 actions)
        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)

def Bench_actions_templerun_nomove(num_frames, num_samples_per_action=4):
    """
    Generate conditioning data for templerun with only nomove action.
    Creates conditioning that reflects no movement in the game.
    """
    actions_single_action = [
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
            if not sub_act in action_name:
                continue
            if sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)