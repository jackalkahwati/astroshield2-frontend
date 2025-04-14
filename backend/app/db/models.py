class Spacecraft(Base):
    """Spacecraft model with proper indexing and validation constraints."""
    __tablename__ = "spacecraft"
    
    id = Column(Integer, primary_key=True, index=True)
    norad_id = Column(String, unique=True, index=True, nullable=False)  # Required field
    cospar_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)  # Required field
    type = Column(String, index=True)
    country = Column(String, index=True)
    launch_date = Column(DateTime)
    operational = Column(Boolean, default=True, index=True)
    orbit_type = Column(String)
    perigee = Column(Float, CheckConstraint('perigee > 0'))  # Must be positive
    apogee = Column(Float, CheckConstraint('apogee > 0'))    # Must be positive
    inclination = Column(Float, CheckConstraint('inclination >= 0 AND inclination <= 180'))  # Valid range
    period = Column(Float, CheckConstraint('period > 0'))    # Must be positive
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Validation constraints and indexes
    __table_args__ = (
        CheckConstraint('apogee >= perigee', 'check_apogee_gt_perigee'),
        Index('idx_spacecraft_norad_id', 'norad_id'),
        Index('idx_spacecraft_type', 'type'),
        Index('idx_spacecraft_country', 'country'),
        Index('idx_spacecraft_operational', 'operational'),
    )
    
    # Relationships with proper back_populates for bidirectional relationships
    conjunctions = relationship("Conjunction", back_populates="spacecraft", 
                               cascade="all, delete-orphan")
    maneuvers = relationship("Maneuver", back_populates="spacecraft",
                            cascade="all, delete-orphan")


class Conjunction(Base):
    """Conjunction event model with validation constraints."""
    __tablename__ = "conjunctions"
    
    id = Column(Integer, primary_key=True, index=True)
    spacecraft_id = Column(Integer, ForeignKey("spacecraft.id", ondelete="CASCADE"), index=True, nullable=False)
    secondary_object_id = Column(String, index=True, nullable=False)
    conjunction_time = Column(DateTime, index=True, nullable=False)
    miss_distance = Column(Float, CheckConstraint('miss_distance >= 0'))  # Cannot be negative
    relative_velocity = Column(Float, CheckConstraint('relative_velocity >= 0'))  # Cannot be negative
    collision_probability = Column(Float, index=True, 
                                  CheckConstraint('collision_probability >= 0 AND collision_probability <= 1'))  # Between 0 and 1
    time_to_closest_approach = Column(Float)
    status = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    spacecraft = relationship("Spacecraft", back_populates="conjunctions")
    
    # Composite indexes and additional constraints
    __table_args__ = (
        Index('idx_conjunction_spacecraft_time', 'spacecraft_id', 'conjunction_time'),
        Index('idx_high_risk_conjunctions', 'spacecraft_id', 'collision_probability', 
              'conjunction_time', postgresql_where=text("collision_probability > 0.0001")),
        # Status must be one of the valid values
        CheckConstraint("status IN ('pending', 'active', 'resolved', 'false_alarm')", 'check_valid_status'),
    )


class HistoricalAnalysis(Base):
    """Historical analysis data with validation constraints."""
    __tablename__ = "historical_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    norad_id = Column(String, index=True, nullable=False)
    analysis_date = Column(DateTime, index=True, nullable=False)
    threat_level = Column(String, index=True, nullable=False)
    analysis_type = Column(String, index=True, nullable=False)
    data = Column(JSON, nullable=False)  # Required field
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    
    # Composite index and validation constraints
    __table_args__ = (
        Index('idx_historical_analysis_date_range', 'norad_id', 'analysis_date'),
        Index('idx_historical_analysis_threat_type', 'norad_id', 'threat_level', 'analysis_type'),
        CheckConstraint("threat_level IN ('none', 'low', 'medium', 'high')", 'check_valid_threat_level'),
        CheckConstraint("analysis_type IN ('historical', 'predictive', 'reactive')", 'check_valid_analysis_type'),
    )


class Maneuver(Base):
    """Spacecraft maneuver model with validation constraints."""
    __tablename__ = "maneuvers"
    
    id = Column(Integer, primary_key=True, index=True)
    spacecraft_id = Column(Integer, ForeignKey("spacecraft.id", ondelete="CASCADE"), index=True, nullable=False)
    conjunction_id = Column(Integer, ForeignKey("conjunctions.id", ondelete="SET NULL"), nullable=True, index=True)
    maneuver_time = Column(DateTime, index=True, nullable=False)
    delta_v = Column(Float, CheckConstraint('delta_v >= 0'), nullable=False)  # Cannot be negative
    direction = Column(String, nullable=False)
    fuel_used = Column(Float, CheckConstraint('fuel_used >= 0'))  # Cannot be negative
    status = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    spacecraft = relationship("Spacecraft", back_populates="maneuvers")
    
    # Validation constraints
    __table_args__ = (
        Index('idx_maneuver_time', 'maneuver_time'),
        Index('idx_maneuver_status', 'status'),
        # Status must be one of the valid values
        CheckConstraint("status IN ('planned', 'executing', 'completed', 'failed', 'cancelled')", 
                       'check_valid_maneuver_status'),
    ) 