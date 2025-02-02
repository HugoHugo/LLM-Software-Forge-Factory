from sqlalchemy import create_engine, Column, Integer, String, JSON, ForeignKey, DateTime, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import enum

Base = declarative_base()

class ComponentStatus(enum.Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    FAILED = "failed"

class Component(Base):
    __tablename__ = 'components'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String)
    status = Column(Enum(ComponentStatus), default=ComponentStatus.PLANNED)
    priority = Column(Integer)
    dependencies = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    implementations = relationship("Implementation", back_populates="component")
    tasks = relationship("Task", back_populates="component")

class Implementation(Base):
    __tablename__ = 'implementations'
    
    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey('components.id'))
    version = Column(String(20))
    code_content = Column(String)
    language = Column(String(50))
    framework = Column(String(50))
    dependencies = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    component = relationship("Component", back_populates="implementations")
    tests = relationship("Test", back_populates="implementation")
    documentation = relationship("Documentation", back_populates="implementation")

class Test(Base):
    __tablename__ = 'tests'
    
    id = Column(Integer, primary_key=True)
    implementation_id = Column(Integer, ForeignKey('implementations.id'))
    test_type = Column(String(50))
    test_content = Column(String)
    coverage_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    implementation = relationship("Implementation", back_populates="tests")

class Documentation(Base):
    __tablename__ = 'documentation'
    
    id = Column(Integer, primary_key=True)
    implementation_id = Column(Integer, ForeignKey('implementations.id'))
    doc_type = Column(String(50))
    content = Column(String)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    implementation = relationship("Implementation", back_populates="documentation")

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey('components.id'))
    task_type = Column(String(50))
    description = Column(String)
    status = Column(String(20), default='pending')
    priority = Column(Integer)
    dependencies = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    component = relationship("Component", back_populates="tasks")

# Database operations
class DBOperations:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)
    
    def get_next_task(self):
        return self.session.query(Task)\
            .filter(Task.status == 'pending')\
            .order_by(Task.priority.desc())\
            .first()
    
    def save_implementation(self, component_id, code_content, test_content, metadata):
        implementation = Implementation(
            component_id=component_id,
            version='0.1',
            code_content=code_content,
            language='python',
            framework='fastapi',
            metadata=metadata
        )
        self.session.add(implementation)
        
        test = Test(
            implementation=implementation,
            test_type='unit',
            test_content=test_content,
            coverage_metrics={'coverage': 0, 'pending_execution': True}
        )
        self.session.add(test)
        self.session.commit()
        return implementation.id
    
    def update_component_status(self, component_id, status):
        component = self.session.query(Component).get(component_id)
        component.status = status
        self.session.commit()
    
    def close(self):
        self.session.close()