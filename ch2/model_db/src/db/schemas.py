import datetime
from typing import Dict, Optional

from pydantic import BaseModel

class ProjectBase(BaseModel):
    project_name: str
    description: Optional[str]

class ProjectCreate(ProjectBase):
    pass

