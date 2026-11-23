import time
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.1.197", is_radian=True)

try:
    print("固件版本：", arm.version)

    for _ in range(20):
        code_pose, pose = arm.get_position(is_radian=True)
        code_joint, joints = arm.get_servo_angle(is_radian=True)
        code_state, state = arm.get_state()
        code_error, errors = arm.get_err_warn_code()

        if code_pose == 0:
            print("末端位姿 [mm, mm, mm, rad, rad, rad]：", pose)
        else:
            print("读取末端失败：", code_pose)

        if code_joint == 0:
            print("关节角 [rad]：", joints[:6])
        else:
            print("读取关节失败：", code_joint)

        print("运行状态：", code_state, state)
        print("错误/警告：", code_error, errors)
        time.sleep(0.2)
finally:
    arm.disconnect()