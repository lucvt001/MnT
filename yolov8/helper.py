import torch
import os

def num_to_range(num, inMin, inMax, outMin, outMax):
    return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def cmd_out(center, box_dim, LeftRightThreshold = 0.25, min_box_sz = 0.04, max_box_sz = 0.36, min_pwm = 100, max_pwm = 255):
    '''
    Process the center and box size of the tracked object and output pwm command.
    
    center: 1-D tensor
    box_dim: 1-D tensor
    LeftRightThreshold: float. The minimum value of the center of the box, above which the bot will start to move left/right.
    min_box_sz: minimum value of the detection box. If area of the detection box is smaller, the bot will start to move forwards.
    max_box_sz: maximum value of the detection box. If area of the detection box is larger, the bot will start to move backwards.
    min_pwm: minimum value to be input into the bot if we want the bot to move (the bot might not move if the pwm input is too low).
    max_pwm: minimum value to be input into the bot if we want the bot to move (control the maximum speed of the bot).

    Convention: Forward and Turn Left are positive pwm values.
    Return (leftRightPwm, forwardBackwardPwm)'''

    leftRightPwm, forwardBackwardPwm = 0, 0
    leftRightDeviation = center[0].item() - 0.5
    box_sz = torch.prod(box_dim).item()

    # if the object is to the right of the camera view
    if leftRightDeviation > LeftRightThreshold:
        leftRightPwm = num_to_range(leftRightDeviation, LeftRightThreshold, 0.5, min_pwm, max_pwm)
    # if the object is to the left of the camera view
    elif leftRightDeviation < -LeftRightThreshold:
        leftRightPwm = num_to_range(leftRightDeviation, -LeftRightThreshold, -0.5, -min_pwm, -max_pwm)

    # if the object is too far away, output positive pwm signal
    if box_sz < min_box_sz:
        forwardBackwardPwm = num_to_range(abs(box_sz - min_box_sz), 0, min_box_sz, min_pwm, max_pwm)
    
    # if the output is too close, output negative pwm signal
    elif box_sz > max_box_sz:
        forwardBackwardPwm = num_to_range(abs(box_sz - max_box_sz), 0, 1-max_box_sz, -min_pwm, -max_pwm)

    return (leftRightPwm, forwardBackwardPwm)

def get_dir(file_name):
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    required_path = os.path.join(current_directory, file_name)
    return required_path

def get_dev():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'