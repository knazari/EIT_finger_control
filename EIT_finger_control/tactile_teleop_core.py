from math import cos, radians, sin
from typing import Dict, Optional


class TactileTeleopCore:
    """
    Converts tactile intent into MG400 jog commands.

    Input command format:
    {
        "active": True/False,
        "mode": "single" or "double",
        "tx": float in [-1, 1],   # translational x intent in sensor frame
        "ty": float in [-1, 1],   # translational y intent in sensor frame
        "rz": float in [-1, 1],   # rotational intent
        "meta": {...}             # optional
    }

    Output format:
    {
        "jog_command": "X+" / "X-" / "Y+" / "Y-" / "R+" / "R-" / "",
        "speed_factor": int,
        "active": bool,
        "mode": str,
        "strength": float,
        "tx": float,              # rotated + filtered x in robot frame
        "ty": float,              # rotated + filtered y in robot frame
        "rz": float,
        "sensor_tx": float,       # original pre-rotation x after preprocessing
        "sensor_ty": float,       # original pre-rotation y after preprocessing
        "offset_deg": float,
    }
    """

    def __init__(
        self,
        deadband: float = 0.08,
        filter_alpha: float = 0.35,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_rz: bool = False,
        normalize_translation: bool = True,
        dominant_axis_ratio: float = 1.15,
        axis_switch_ratio: float = 1.35,
        min_speed_factor: int = 10,
        max_speed_factor: int = 35,
        rotation_min_speed_factor: int = 10,
        rotation_max_speed_factor: int = 30,
        sensor_angle_offset_deg: float = 0.0,
        rotation_hold_frames: int = 4,
    ) -> None:
        self.deadband = float(deadband)
        self.filter_alpha = float(filter_alpha)

        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_rz = invert_rz
        self.normalize_translation = normalize_translation

        # Sensor-frame to robot-frame rotation offset
        self.sensor_angle_offset_deg = float(sensor_angle_offset_deg)

        # Dominant axis selection
        self.dominant_axis_ratio = float(dominant_axis_ratio)

        # Extra hysteresis when switching from X <-> Y
        # New axis must be stronger by this ratio before switching.
        self.axis_switch_ratio = float(axis_switch_ratio)

        self.min_speed_factor = int(min_speed_factor)
        self.max_speed_factor = int(max_speed_factor)
        self.rotation_min_speed_factor = int(rotation_min_speed_factor)
        self.rotation_max_speed_factor = int(rotation_max_speed_factor)

        # Small persistence for rotation direction
        self.rotation_hold_frames = int(max(0, rotation_hold_frames))

        # Internal filtered state
        self.filtered_tx = 0.0
        self.filtered_ty = 0.0
        self.filtered_rz = 0.0

        # Hysteresis / state memory
        self.last_translation_jog_command: str = ""
        self.last_rotation_jog_command: str = ""
        self.rotation_hold_counter: int = 0

    def set_sensor_angle_offset_deg(self, offset_deg: float) -> None:
        """
        Update the sensor-to-robot angular offset.
        This lets you re-align the finger after remounting it.
        """
        self.sensor_angle_offset_deg = float(offset_deg)

    def reset_filters(self) -> None:
        self.filtered_tx = 0.0
        self.filtered_ty = 0.0
        self.filtered_rz = 0.0
        self.last_translation_jog_command = ""
        self.last_rotation_jog_command = ""
        self.rotation_hold_counter = 0

    def apply_deadband(self, value: float) -> float:
        if abs(value) < self.deadband:
            return 0.0
        return value

    @staticmethod
    def clip_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def low_pass_filter(self, new_value: float, old_value: float) -> float:
        alpha = self.filter_alpha
        return alpha * new_value + (1.0 - alpha) * old_value

    def rotate_translation_to_robot_frame(self, tx: float, ty: float) -> tuple[float, float]:
        """
        Rotate translation vector from sensor frame to robot frame.
        """
        theta = radians(self.sensor_angle_offset_deg)
        c = cos(theta)
        s = sin(theta)

        tx_rot = c * tx - s * ty
        ty_rot = s * tx + c * ty
        return tx_rot, ty_rot

    def preprocess_command(self, tactile_cmd: Dict) -> Dict:
        active = bool(tactile_cmd.get("active", False))
        mode = tactile_cmd.get("mode", "single")

        tx = self.clip_unit(tactile_cmd.get("tx", 0.0))
        ty = self.clip_unit(tactile_cmd.get("ty", 0.0))
        rz = self.clip_unit(tactile_cmd.get("rz", 0.0))

        tx = self.apply_deadband(tx)
        ty = self.apply_deadband(ty)
        rz = self.apply_deadband(rz)

        if self.invert_x:
            tx = -tx
        if self.invert_y:
            ty = -ty
        if self.invert_rz:
            rz = -rz

        if mode == "single" and self.normalize_translation:
            mag = (tx ** 2 + ty ** 2) ** 0.5
            if mag > 1.0:
                tx /= mag
                ty /= mag

        return {
            "active": active,
            "mode": mode,
            "tx": tx,   # still in sensor frame here
            "ty": ty,   # still in sensor frame here
            "rz": rz,
        }

    @staticmethod
    def _map_strength_to_speed(
        strength: float,
        min_speed: int,
        max_speed: int,
    ) -> int:
        strength = max(0.0, min(1.0, float(strength)))
        speed = min_speed + strength * (max_speed - min_speed)
        return int(round(speed))

    @staticmethod
    def _axis_of_jog_command(jog_command: str) -> Optional[str]:
        if jog_command.startswith("X"):
            return "X"
        if jog_command.startswith("Y"):
            return "Y"
        if jog_command.startswith("R"):
            return "R"
        return None

    def _raw_translation_to_jog(self, tx: float, ty: float) -> str:
        """
        Convert translation intent in robot frame into a dominant-axis jog command,
        without hysteresis.
        """
        abs_tx = abs(tx)
        abs_ty = abs(ty)

        if abs_tx == 0.0 and abs_ty == 0.0:
            return ""

        if abs_tx > self.dominant_axis_ratio * abs_ty:
            return "X+" if tx > 0 else "X-"

        if abs_ty > self.dominant_axis_ratio * abs_tx:
            # Keep the current convention from your setup
            return "Y-" if ty > 0 else "Y+"

        if abs_tx >= abs_ty:
            return "X+" if tx > 0 else "X-"
        else:
            return "Y-" if ty > 0 else "Y+"

    def _translation_to_jog(self, tx: float, ty: float) -> str:
        """
        Convert filtered translation intent in robot frame into a jog command
        with X/Y switching hysteresis.
        """
        candidate = self._raw_translation_to_jog(tx, ty)

        if candidate == "":
            self.last_translation_jog_command = ""
            return ""

        if self.last_translation_jog_command == "":
            self.last_translation_jog_command = candidate
            return candidate

        prev_axis = self._axis_of_jog_command(self.last_translation_jog_command)
        cand_axis = self._axis_of_jog_command(candidate)

        if prev_axis == cand_axis:
            # Same axis, allow sign to change immediately on that axis
            self.last_translation_jog_command = candidate
            return candidate

        # Axis switch hysteresis
        abs_tx = abs(tx)
        abs_ty = abs(ty)

        if prev_axis == "X" and cand_axis == "Y":
            # Require Y to be clearly stronger before switching from X to Y
            if abs_ty > self.axis_switch_ratio * max(abs_tx, 1e-6):
                self.last_translation_jog_command = candidate
                return candidate
            return self.last_translation_jog_command

        if prev_axis == "Y" and cand_axis == "X":
            # Require X to be clearly stronger before switching from Y to X
            if abs_tx > self.axis_switch_ratio * max(abs_ty, 1e-6):
                self.last_translation_jog_command = candidate
                return candidate
            return self.last_translation_jog_command

        self.last_translation_jog_command = candidate
        return candidate

    def _rotation_to_jog(self, rz: float) -> str:
        """
        Rotation mapping with small hold to prevent rapid flicker.
        """
        if abs(rz) <= 0.0:
            self.last_rotation_jog_command = ""
            self.rotation_hold_counter = 0
            return ""

        candidate = "R+" if rz > 0 else "R-"

        if self.last_rotation_jog_command == "":
            self.last_rotation_jog_command = candidate
            self.rotation_hold_counter = self.rotation_hold_frames
            return candidate

        if candidate == self.last_rotation_jog_command:
            self.rotation_hold_counter = self.rotation_hold_frames
            return candidate

        # Opposite direction requested
        if self.rotation_hold_counter > 0:
            self.rotation_hold_counter -= 1
            return self.last_rotation_jog_command

        self.last_rotation_jog_command = candidate
        self.rotation_hold_counter = self.rotation_hold_frames
        return candidate

    def compute_jog_command(self, tactile_cmd: Dict) -> Dict:
        """
        Returns:
        {
            "jog_command": str,
            "speed_factor": int,
            "active": bool,
            "mode": str,
            "strength": float,
            "tx": float,          # rotated + filtered x in robot frame
            "ty": float,          # rotated + filtered y in robot frame
            "rz": float,
            "sensor_tx": float,   # original pre-rotation x after preprocessing
            "sensor_ty": float,   # original pre-rotation y after preprocessing
            "offset_deg": float,
        }
        """
        cmd = self.preprocess_command(tactile_cmd)

        if not cmd["active"]:
            self.reset_filters()
            return {
                "jog_command": "",
                "speed_factor": self.min_speed_factor,
                "active": False,
                "mode": cmd["mode"],
                "strength": 0.0,
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "sensor_tx": 0.0,
                "sensor_ty": 0.0,
                "offset_deg": self.sensor_angle_offset_deg,
            }

        mode = cmd["mode"]

        if mode == "single":
            sensor_tx = cmd["tx"]
            sensor_ty = cmd["ty"]

            # Rotate from sensor frame into robot frame
            rot_tx, rot_ty = self.rotate_translation_to_robot_frame(sensor_tx, sensor_ty)

            # Filter in robot frame
            self.filtered_tx = self.low_pass_filter(rot_tx, self.filtered_tx)
            self.filtered_ty = self.low_pass_filter(rot_ty, self.filtered_ty)
            self.filtered_rz = 0.0
            self.last_rotation_jog_command = ""
            self.rotation_hold_counter = 0

            strength = min(1.0, (self.filtered_tx ** 2 + self.filtered_ty ** 2) ** 0.5)
            jog_command = self._translation_to_jog(self.filtered_tx, self.filtered_ty)
            speed_factor = self._map_strength_to_speed(
                strength,
                self.min_speed_factor,
                self.max_speed_factor,
            )

            return {
                "jog_command": jog_command,
                "speed_factor": speed_factor,
                "active": jog_command != "",
                "mode": "single",
                "strength": strength,
                "tx": self.filtered_tx,
                "ty": self.filtered_ty,
                "rz": 0.0,
                "sensor_tx": sensor_tx,
                "sensor_ty": sensor_ty,
                "offset_deg": self.sensor_angle_offset_deg,
            }

        if mode == "double":
            self.filtered_tx = 0.0
            self.filtered_ty = 0.0
            self.last_translation_jog_command = ""

            self.filtered_rz = self.low_pass_filter(cmd["rz"], self.filtered_rz)

            strength = min(1.0, abs(self.filtered_rz))
            jog_command = self._rotation_to_jog(self.filtered_rz)
            speed_factor = self._map_strength_to_speed(
                strength,
                self.rotation_min_speed_factor,
                self.rotation_max_speed_factor,
            )

            return {
                "jog_command": jog_command,
                "speed_factor": speed_factor,
                "active": jog_command != "",
                "mode": "double",
                "strength": strength,
                "tx": 0.0,
                "ty": 0.0,
                "rz": self.filtered_rz,
                "sensor_tx": 0.0,
                "sensor_ty": 0.0,
                "offset_deg": self.sensor_angle_offset_deg,
            }

        self.reset_filters()
        return {
            "jog_command": "",
            "speed_factor": self.min_speed_factor,
            "active": False,
            "mode": mode,
            "strength": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "rz": 0.0,
            "sensor_tx": 0.0,
            "sensor_ty": 0.0,
            "offset_deg": self.sensor_angle_offset_deg,
        }