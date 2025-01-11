from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Spacecraft(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class CCDMIndicator(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    spacecraft_id = db.Column(db.Integer, db.ForeignKey('spacecraft.id'), nullable=False)
    conjunction_type = db.Column(db.String(50))
    relative_velocity = db.Column(db.Float)
    miss_distance = db.Column(db.Float)
    time_to_closest_approach = db.Column(db.Float)
    probability_of_collision = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'spacecraft_id': self.spacecraft_id,
            'conjunction_type': self.conjunction_type,
            'relative_velocity': self.relative_velocity,
            'miss_distance': self.miss_distance,
            'time_to_closest_approach': self.time_to_closest_approach,
            'probability_of_collision': self.probability_of_collision,
            'timestamp': self.timestamp.isoformat()
        }
