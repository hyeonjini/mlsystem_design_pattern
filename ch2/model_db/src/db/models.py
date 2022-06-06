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
     __tablename__ = "projects"

     project_id = Column(
         String(255),
         primary_key=True,
         comment="basic_key"
     )

     project_name = Column(
         String(255),
         nullable=False,
         unique=True,
         comment="project_name"
     )

     description = Column(
         Text,
         nullable=True,
         comment="description"
     )

     created_datetime = Column(
         DateTime(timezone=True),
         sever_default=current_timestamp(),
         nuallable=False,
     )

class Model(Base):
    __tablename__ = "models"

    model_id = Column(
        String(255),
        primary_key=True,
        comment="basic_key"
    )

    project_id = Column(
        String(255),
        ForeignKey("projects.project_id"),
        nullable=False,
        comment="foreign_key"
    )

    model_name = Column(
        String(255),
        nullable=False,
        comment="model_name"
    )

    description = Column(
        Text,
        nullable=True,
        comment="description"
    )

    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False
    )

class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(
        String(255),
        primary_key=True,
        comment="basic_key"
    )

    model_version_id = Column(
        String(255),
        nullable=False,
        comment="model_experiment_version"
    )

    model_id = Column(
        String(255),
        ForeignKey("models.model_id"),
        nullable=False,
        comment="foreign_key"
    )

    parameters = Column(
        JSON,
        nullable=True,
        comment="training_parameters"
    )

    training_dataset = Column(
        Text,
        nullable=True,
        comment="training_dataset"
    )

    validation_dataset = Column(
        Text,
        nullable=True,
        comment="validation_dataset"
    )

    test_dataset = Column(
        Text,
        nullable=True,
        comment="test_dataset"
    )

    evaluations = Column(
        JSON,
        nullable=True,
        comment="evaluations"
    )

    artifact_file_paths = Column(
        JSON,
        nullable=True,
        comment="artifact_file_paths"
    )

    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False
    )