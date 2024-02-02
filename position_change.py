from interbotix_xs_modules.arm import InterbotixManipulatorXS

bot = InterbotixManipulatorXS("wx250", moving_time=1.5, accel_time=0.75)

bot.arm.set_ee_pose_components(x=0.3, y=0.1, z=0.2)