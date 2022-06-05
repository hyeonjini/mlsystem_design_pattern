import uuid

from typing import Dict, List, Optional

from sqlalchemy.orm import Session
from src.db import models, schemas


def select_project_all(db: Session) -> List[schemas.Project]:
    return db.query(models.Project).all()

def select_project_by_id(db: Session, project_id: str) -> schemas.Project:
    return db.query(models.Project).filter(models.Project.project_id == project_id).first()

def select_project_by_name(db: Session, project_name: str) -> schemas.Project:
    return db.query(models.Project).filter(models.Project.project_name == project_name).first()

def add_project(
        db: Session, 
        project_name: str, 
        description: Optional[str] = None, 
        commit: bool = True
    ) -> schemas.Project:

    exists = select_project_by_name(
        db=db,
        project_name=project_name
    )

    if exists:
        return exists
    
    else:
        project_id = str(uuid.uuid4())[:6]
        data = models.Project(
            project_id=project_id,
            project_name=project_name,
            description=description
        )
        db.add(data)
        if commit:
            db.commit()
            db.refresh(data)
        return data

def select_model_all(db: Session) -> List[schemas.Model]:
    return db.query(models.Model).all()

def select_model_by_id(
    db: Session,
    model_id: str,
) -> schemas.Model:
    return db.query(models.Model).filter(models.Model.model_id == model_id).first()

