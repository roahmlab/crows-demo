import torch
import argparse
import numpy as np
import time
import json
import random
from tqdm import tqdm
from environments.urdf_obstacle import KinematicUrdfWithObstacles
from environments.fullstep_recorder import FullStepRecorder
from planning.sparrows.sparrows_urdf import SPARROWS_3D_planner
from planning.crows.crows_urdf import CROWS_3D_planner
from planning.common.waypoints import GoalWaypointGenerator, CustomWaypointGenerator
from visualizations.sphere_viz import SpherePlannerViz
import os
import csv

T_PLAN, T_FULL = 0.5, 1.0


def evaluate_planner(planner, 
                     n_steps=150, 
                     video=False, 
                     reachset_viz=False, 
                     time_limit=0.5, 
                     t_final_thereshold=0.,
                     check_self_collision=False,
                     tol = 1e-5,
                    ):


    ############  NOTE ############
    obs_pos = []
    obs_size = []
    configs = []
    with open('living_room/living_room_obstacles.csv', mode ='r') as file:   
        csvFile = csv.reader(file)
        for line in csvFile:
                obs_pos.append([float(num) for num in line][:3])
                obs_size.append([float(num) for num in line][3:6])
    with open('living_room/living_room_config.csv', mode ='r') as file:   
        csvFile = csv.reader(file)
        for line in csvFile:
                configs.append([float(num) for num in line][:7])
    
    configs = np.array(configs)
    configs = (configs + np.pi) % (2 * np.pi) - np.pi

    obs_pos = obs_pos[:-1]
    obs_size = obs_size[:-1]
    theta = [0]*len(obs_pos)

    qstart = configs[0]
    qgoal = configs[-1]
    waypoints = configs[1:]

    #theta = [np.pi/180*(-30), np.pi/180*(60)] # Rotation angles of the obstacles w.r.t. the z-axis, shape (n_obs,)
    #obs_pos = [[0.5,0.2,0.1], [-0.5,0.2,0.1]] # Positions of the obstacles as rows of xyz coordinates, shape (n_obs, 3)
    #obs_size = [[0.5,0.2,0.2], [0.8,0.1,0.2]] # Sizes of the obstacles as rows of xyz dimensions, shape (n_obs, 3)
    
    # qstart = np.array([2.796017461694916, 0.6283185307179586, 0.15707963267948966, 0.6785840131753953, -0.3141592653589793, 0.7005751617505239, 0]) # Initial configuration, shape (7,)
    # qgoal = np.array([2.7991590543485056, 0.38327430373795474, 0.15707963267948966, 0.47752208334564855, -0.3141592653589793, 0.6723008278682158, 0]) # Goal configuration, shape (7,)
    # waypoints = None  # Optional: Intermediate waypoints from the initial to goal configuration, shape (num_waypoints, 7)
    # NOTE: len(theta), len(obs_pos), len(obs_size) should match
    ############  NOTE ############


    save_folder = 'planning_demo'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if video:
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'


    n_obs = len(theta)
    obs_rot = [np.array([[np.cos(th),-np.sin(th),0],[np.sin(th), np.cos(th), 0],[0,0,1]]) for th in theta]
    trajectory = {'qpos':[],'qvel':[],'ka':[]}
    env = KinematicUrdfWithObstacles(
            robot=rob.urdf,
            step_type='integration',
            check_joint_limits=True,
            check_self_collision=check_self_collision,
            use_bb_collision=False,
            render_mesh=True,
            reopen_on_close=False,
            obs_size_min = [0.01,0.01,0.01],
            obs_size_max = [0.5,0.5,0.5],
            n_obs=n_obs,
            renderer = 'pyrender-offscreen',
            info_nearest_obstacle_dist = False,
            obs_gen_buffer = 0.01
        )
    if video and reachset_viz:
        viz = SpherePlannerViz(planner, plot_full_set=True, t_full=T_FULL)
        env.add_render_callback('spheres', viz.render_callback, needs_time=False)

    
    obs = env.reset(
                qpos = qstart, 
                qvel = np.zeros_like(qstart), 
                qgoal = qgoal, 
                obs_pos = obs_pos,
                obs_size = obs_size,
                obs_rot = obs_rot,
    )

    if waypoints is None:
        waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*3)
    else:
        waypoint_generator = CustomWaypointGenerator(waypoints, obs['qgoal'], planner.osc_rad*3)

    if video:
        video_path = os.path.join(save_folder, 'video.mp4')
        video_recorder = FullStepRecorder(env, path=video_path)

    force_fail_safe = False 
    was_stuck = False   
    for _ in range(n_steps):
        qpos, qvel, qgoal = obs['qpos'], obs['qvel'], obs['qgoal']
        obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']), np.asarray(obs['obstacle_rot']))
        waypoint = waypoint_generator.get_waypoint(qpos, qvel)
        ka, flag, planner_stat = planner.plan(qpos, qvel, waypoint, obstacles, time_limit=time_limit, t_final_thereshold=t_final_thereshold, tol=tol)   


        if flag != 0:
            ka = (0 - qvel)/(T_FULL - T_PLAN)

        if force_fail_safe:
            ka = (0 - qvel)/(T_FULL - T_PLAN)
            force_fail_safe = False   
        else:
            force_fail_safe = (flag == 0) and planner.nlp_problem_obj.use_t_final and (np.sqrt(planner.final_cost) < env.goal_threshold)

        trajectory['qpos'].append(qpos)
        trajectory['qvel'].append(qvel)
        trajectory['ka'].append(ka)

        if video and reachset_viz:
            if flag == 0:
                viz.set_ka(ka)
            else:
                viz.set_ka(None)

        obs, reward, done, info = env.step(ka)


        if video:
            video_recorder.capture_frame()

        if done:
            trajectory['qpos'].append(obs['qpos'])
            trajectory['qvel'].append(obs['qvel'])

            if info['collision_info']['in_collision']:
                print('Collision!')
            elif reward == 1:
                print('Success')
            else:
                print('Terminated')
            break

        if flag != 0:
            if was_stuck:
                print('Stuck!')
                break
            else:
                was_stuck = True
        else:
            was_stuck = False


        

    if video:
        video_recorder.close()
    
    trajectory['qpos'] = np.asarray(trajectory['qpos']).tolist()
    trajectory['qvel'] = np.asarray(trajectory['qvel']).tolist()
    trajectory['ka'] = np.asarray(trajectory['ka']).tolist()
    
    with open(os.path.join(save_folder,'trajectory.json'), 'w') as f:
        json.dump(trajectory, f, indent=2)       
    ############  NOTE ############
    # The trajectory will be saved as a sequence of joint configurations, joint velocities, and trajectory parameters at every 0.5-second.
    
    # qpos, shape (n_steps + 1, 7) -> [p_0, p_1, ..., p_N]  # Joint positions at each time step
    # qvel, shape (n_steps + 1, 7) -> [v_0, v_1, ..., v_N]  # Joint velocities at each time step
    # ka, shape (n_steps, 7)       -> [ka_0, ka_1, ..., ka_{N-1}]  # Trajectory parameters (acceleration) for each time step
    
    ############  NOTE ############


def read_params():
    parser = argparse.ArgumentParser(description="Arm Planning")
    # general setting
    parser.add_argument('--n_steps', type=int, default=150)
    parser.add_argument('--device', type=int, default=0 if torch.cuda.is_available() else -1, choices=range(-1,torch.cuda.device_count())) # Designate which cuda to use, default: cpu

    # visualization settings
    parser.add_argument('--video',  action='store_true')
    parser.add_argument('--reachset',  action='store_true')
        
    # optimization info
    parser.add_argument('--num_spheres', type=int, default=5)
    parser.add_argument('--time_limit',  type=float, default=1e20)
    parser.add_argument('--t_final_thereshold', type=float, default=0.2)
    parser.add_argument('--solver', type=str, default="ma27")
    parser.add_argument('--tol', type=float, default=1e-3) # desired convergence tolerance for IPOPT solver

   # CROWS
    parser.add_argument('--not_use_learned_grad', action='store_true')  # whether to not use learned gradient for CROWS
    parser.add_argument('--confidence_idx', type=int, default=2) #option for confidence level of CROWS model uncertainty -> {idx:epsilon_hat}, 0: 99.999%, 1: 99.99%, 2: 99.9%, 3: 99% 4: 90% 5:80%
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    
    params = read_params()
    
    # Set device
    device = torch.device('cpu') if params.device <0 else torch.device(f'cuda:{params.device}')
    # Set dtype
    dtype = torch.float32 

    print(f"Running CROWS 3D7Links with {params.n_steps} step limit and {params.time_limit}s time limit each step")
    print(f"Using device {device}")
        
    import zonopyrobots as robots2
    robots2.DEBUG_VIZ = False
    basedirname = os.path.dirname(robots2.__file__)
    robot_path = 'robots/assets/robots/kinova_arm/gen3.urdf'
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, robot_path), dtype = dtype, device=device, create_joint_occupancy=True)

    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')

    joint_radius_override = {
            'joint_1': torch.tensor(0.0503305, dtype=torch.float, device=device),
            'joint_2': torch.tensor(0.0630855, dtype=torch.float, device=device),
            'joint_3': torch.tensor(0.0463565, dtype=torch.float, device=device),
            'joint_4': torch.tensor(0.0634475, dtype=torch.float, device=device),
            'joint_5': torch.tensor(0.0352165, dtype=torch.float, device=device),
            'joint_6': torch.tensor(0.0542545, dtype=torch.float, device=device),
            'joint_7': torch.tensor(0.0364255, dtype=torch.float, device=device),
            'end_effector': torch.tensor(0.0394685, dtype=torch.float, device=device),
        }

    planner = CROWS_3D_planner(
        rob, 
        dtype = dtype,
        device=device, 
        sphere_device=device, 
        spheres_per_link=params.num_spheres,
        joint_radius_override=joint_radius_override,
        linear_solver=params.solver,
        model_dir = model_dir,
        use_learned_grad = not params.not_use_learned_grad,
        confidence_idx = params.confidence_idx
    )

    # planner = SPARROWS_3D_planner(
    #     rob, 
    #     dtype = dtype,
    #     device=device, 
    #     sphere_device=device, 
    #     spheres_per_link=params.num_spheres,
    #     joint_radius_override=joint_radius_override,
    #     linear_solver=params.solver,
    # )

    evaluate_planner(
        planner=planner, 
        n_steps=params.n_steps, 
        video=params.video,
        reachset_viz=params.reachset,
        time_limit=params.time_limit,
        t_final_thereshold=params.t_final_thereshold,
        tol = params.tol,
    )