"""Logger for QuadSwarm inside HARL."""

from harl.common.base_logger import BaseLogger


class QuadSwarmLogger(BaseLogger):
    def get_task_name(self):
        obstacle = "obstacle" if self.env_args.get("use_obstacles", False) else "no_obstacle"
        return (
            f"{self.env_args.get('quads_mode', 'static_same_goal')}_"
            f"{self.env_args.get('num_agents', 4)}agents_"
            f"{obstacle}"
        )

