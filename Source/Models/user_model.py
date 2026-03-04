from dataclasses import dataclass

ROLE_SYSADMIN    = "sysadmin"
ROLE_RADIOLOGIST = "radiologist"
ROLE_XRAY_TECH   = "xray_tech"
ROLE_SCHEDULER   = "scheduler"

ALL_ROLES = [ROLE_SYSADMIN, ROLE_RADIOLOGIST, ROLE_XRAY_TECH, ROLE_SCHEDULER]

ROLE_DISPLAY = {
    ROLE_SYSADMIN:    "System Administrator",
    ROLE_RADIOLOGIST: "Radiologist",
    ROLE_XRAY_TECH:   "X-Ray Technician",
    ROLE_SCHEDULER:   "Scheduling Assistant",
}

PERMISSIONS = {
    ROLE_SYSADMIN: {
        "can_upload",
        "can_diagnose",
        "can_view_scans",
        "can_manage_accounts",
        "can_view_all_patients",
        "can_manage_schedule",
    },
    ROLE_RADIOLOGIST: {
        "can_upload",
        "can_diagnose",
        "can_view_scans",
        "can_view_all_patients",
    },
    ROLE_XRAY_TECH: {
        "can_upload",
        "can_view_scans",
    },
    ROLE_SCHEDULER: {
        "can_manage_schedule",
    },  # no scan/patient access
}


@dataclass
class UserRecord:
    username:      str
    role:          str
    first_name:    str
    last_name:     str
    email:         str  = ""
    password_hash: str  = ""
    id:            int  = None
    created_at:    str  = ""
    last_login:    str  = ""
    is_active:     bool = True

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def role_display(self):
        return ROLE_DISPLAY.get(self.role, self.role)

    def has_permission(self, permission: str) -> bool:
        return permission in PERMISSIONS.get(self.role, set())