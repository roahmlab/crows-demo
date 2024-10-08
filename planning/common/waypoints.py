import numpy as np

def wrap_to_pi( configs):
    wrapped_configs = np.mod(configs + np.pi, 2 * np.pi) - np.pi
    return wrapped_configs

class ArmWaypoint():
    __slots__ = ['pos', 'vel']
    def __init__(self, pos, vel = None):
        self.pos = pos
        self.vel = vel

class GoalWaypointGenerator():
    def __init__(self, qgoal, enforce_vel_radius=0):
        self.qgoal = qgoal
        self.enforce_vel_radius = enforce_vel_radius
    
    def get_waypoint(self, qpos, qvel, qgoal=None):
        if qgoal is not None:
            self.qgoal = qgoal
        if np.all(np.abs(qpos - self.qgoal) < self.enforce_vel_radius):
            print("Adding velocity to waypoint")
            return ArmWaypoint(self.qgoal, np.zeros_like(qvel))
        else:
            return ArmWaypoint(self.qgoal)
        
class CustomWaypointGenerator():
    def __init__(
        self,
        waypoints,
        qgoal,
        enforce_vel_radius=0,
        threshold = 0.5,
    ):
        self.traj = waypoints
        self.num_waypoints = self.traj.shape[0]
        self.waypoint_i: int = 1 # the 0-index waypoint is the starting configuration
        self.counter: int = 0
        self.enforce_vel_radius = enforce_vel_radius
        self.qgoal = qgoal
        self.threshold = threshold

    def get_waypoint(self, qpos, qvel):    
        if self.waypoint_i < self.num_waypoints - 1:
            if np.linalg.norm(qpos - self.traj[self.waypoint_i]) < self.threshold:
                self.waypoint_i += 1
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= 2:
                    self.counter = 0
                    self.waypoint_i += 1
            if self.waypoint_i >= self.num_waypoints - 1:
                waypoint = self.qgoal
            else:
                waypoint = self.traj[self.waypoint_i]
        else:
            waypoint = self.qgoal
        
        if np.all(np.abs(qpos - waypoint) < self.enforce_vel_radius):
            print("Adding velocity to waypoint")
            return ArmWaypoint(waypoint, np.zeros_like(qvel))
        else:
            return ArmWaypoint(waypoint)


class HackWaypointGenerator():
    def __init__(
        self,
        waypoints,
        qgoal,
        enforce_vel_radius=0,
        joint_waypoint_tolerance = 0.01,
    ):
        self.traj = waypoints
        self.num_waypoints = self.traj.shape[0]
        self.counter: int = 0

        self.traj_list_rev = np.vstack((waypoints, qgoal)).tolist()
        self.traj_list_rev.reverse()
        self.joint_waypoint_tolerance = joint_waypoint_tolerance

        self.qgoal = None 

    def update_waypoint(self):
        if len(self.traj_list_rev)>0:
            self.qgoal = np.array(self.traj_list_rev.pop())

    def get_waypoint(self, qpos, qvel):   

        if self.qgoal is None:
            self.update_waypoint()

        if len(self.traj_list_rev) > 1:
            future_goals = np.array(self.traj_list_rev)
            errs = np.abs(wrap_to_pi(qpos-future_goals))

            close_indices = np.all(errs < self.joint_waypoint_tolerance , axis = 1)

            if np.any(close_indices):
                first_close_idx = np.where(close_indices)[0][0] 
                self.traj_list_rev = self.traj_list_rev[:first_close_idx]
                self.update_waypoint()

        errs = np.abs(wrap_to_pi(qpos-self.qgoal))
        if np.all(errs < self.joint_waypoint_tolerance):
            self.update_waypoint()

        errs = np.abs(wrap_to_pi(qpos-self.qgoal))
        if len(self.traj_list_rev) > 0 or np.any(errs >= self.joint_waypoint_tolerance):
            return ArmWaypoint(self.qgoal)
        else:
            print("Adding velocity to waypoint")
            return ArmWaypoint(self.qgoal, np.zeros_like(qvel))
