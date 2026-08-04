from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
import datetime
from .database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    client = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    project_type = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    progress = Column(Integer, default=0, nullable=False)  # 0-100, updated by agent tasks
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    team_members = relationship("TeamMember", back_populates="project", cascade="all, delete-orphan")


class TeamMember(Base):
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # developer, pm, designer, qa, devops

    project = relationship("Project", back_populates="team_members")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    file_path = Column(String, nullable=False)
    filename = Column(String)
    summary = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="documents")


# ---------------------------------------------------------------------------
# Seed: equipo dummy disponible para asignar a cualquier proyecto
# ---------------------------------------------------------------------------
DUMMY_TEAM_MEMBERS = [
    {"name": "Ernesto Muñoz Nieves",   "role": "pm"},
    {"name": "Oscar Fragoso",    "role": "developer"},
    {"name": "Eric David Guadarrama",    "role": "developer"},
    {"name": "Eric Zurita",       "role": "developer"},
    {"name": "Jorge Maza",     "role": "developer"},
    {"name": "Wilfrido Castillo",    "role": "pm"},
    {"name": "Alejandro Perez",   "role": "pm"},
    
]

