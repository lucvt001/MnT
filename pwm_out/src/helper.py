def num_to_range(num, inMin, inMax, outMin, outMax):
    return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def cmd_out(center_x, height, LeftRightThreshold = 0.25, min_height = 0.5, max_height = 0.8, min_pwm = 100, max_pwm = 255):
    '''
    Process the center_x and box size of the tracked object and output pwm command.
    
    center_x: float
    height: height of bounding box
    LeftRightThreshold: float. The minimum value of the center_x of the box, above which the bot will start to move left/right.
    min_height: minimum value of the detection box. If area of the detection box is smaller, the bot will start to move forwards.
    max_height: maximum value of the detection box. If area of the detection box is larger, the bot will start to move backwards.
    min_pwm: minimum value to be input into the bot if we want the bot to move (the bot might not move if the pwm input is too low).
    max_pwm: minimum value to be input into the bot if we want the bot to move (control the maximum speed of the bot).

    Convention: Forward and Turn Right are positive pwm values.
    Return (leftRightPwm, forwardBackwardPwm)'''

    leftRightPwm, forwardBackwardPwm = 0, 0
    leftRightDeviation = center_x - 0.5

    # if the object is to the right of the camera view
    if leftRightDeviation > LeftRightThreshold:
        leftRightPwm = num_to_range(leftRightDeviation, LeftRightThreshold, 0.5, min_pwm, max_pwm)
    # if the object is to the left of the camera view
    elif leftRightDeviation < -LeftRightThreshold:
        leftRightPwm = num_to_range(leftRightDeviation, -LeftRightThreshold, -0.5, -min_pwm, -max_pwm)

    else:
        leftRightPwm = 0

    # if the object is too far away, output positive pwm signal
    if height < min_height:
        forwardBackwardPwm = num_to_range(abs(height - min_height), 0.4, min_height, min_pwm, max_pwm)
    
    # if the output is too close, output negative pwm signal
    elif height > max_height:
        forwardBackwardPwm = num_to_range(abs(height - max_height), 0, 1-max_height, -min_pwm, -max_pwm)

    else:
        forwardBackwardPwm = 0

    return (leftRightPwm, forwardBackwardPwm)