from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
)

from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from src.db.database import Base

class Project(Base):
     __tablename__ == "projects"

     project_id = Column(
         String(255),
         primary_key=True,
         commnet="Basic_key"
     )

     project_name = Column(
         String(255),
         nullable=False,
         unique=True,
         commenct="Project name"
     )

     description = Column(
         Text,
         nullalble=True,
         commenct="Description"
     )

     created_datetime = Column(
         DateTime(timezone=True),
         sever_default=current_timestamp(),
         nuallable=False,
     )
