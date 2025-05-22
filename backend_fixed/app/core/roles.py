from enum import Enum

class Roles(str, Enum):
    viewer = "viewer"   # basic authenticated user
    analyst = "analyst" # example role for analytical endpoints
    operator = "operator" # can execute maneuvers
    admin = "admin"     # superuser
    system_admin = "system_admin" 