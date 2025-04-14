"""
Migration script to add indexes to improve query performance.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_add_indexes'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """
    Upgrade database schema by adding necessary indexes.
    """
    # Add index on norad_id in spacecraft table
    op.create_index(
        'idx_spacecraft_norad_id', 
        'spacecraft', 
        ['norad_id'], 
        unique=True
    )
    
    # Add index on spacecraft type
    op.create_index(
        'idx_spacecraft_type',
        'spacecraft',
        ['type']
    )
    
    # Add index on spacecraft country
    op.create_index(
        'idx_spacecraft_country',
        'spacecraft',
        ['country']
    )
    
    # Add index on spacecraft operational status
    op.create_index(
        'idx_spacecraft_operational',
        'spacecraft',
        ['operational']
    )
    
    # Conjunction table indexes
    op.create_index(
        'idx_conjunction_secondary_object_id',
        'conjunctions',
        ['secondary_object_id']
    )
    
    op.create_index(
        'idx_conjunction_time',
        'conjunctions',
        ['conjunction_time']
    )
    
    op.create_index(
        'idx_conjunction_probability',
        'conjunctions',
        ['collision_probability']
    )
    
    op.create_index(
        'idx_conjunction_status',
        'conjunctions',
        ['status']
    )
    
    # Composite index for common queries on conjunctions
    op.create_index(
        'idx_conjunction_spacecraft_time',
        'conjunctions',
        ['spacecraft_id', 'conjunction_time']
    )
    
    # Composite index for high-risk conjunctions
    op.execute(
        """
        CREATE INDEX idx_high_risk_conjunctions 
        ON conjunctions (spacecraft_id, collision_probability, conjunction_time) 
        WHERE collision_probability > 0.0001
        """
    )
    
    # Historical analysis indexes
    op.create_index(
        'idx_historical_analysis_norad_id',
        'historical_analyses',
        ['norad_id']
    )
    
    op.create_index(
        'idx_historical_analysis_date',
        'historical_analyses',
        ['analysis_date']
    )
    
    op.create_index(
        'idx_historical_analysis_threat',
        'historical_analyses',
        ['threat_level']
    )
    
    op.create_index(
        'idx_historical_analysis_type',
        'historical_analyses',
        ['analysis_type']
    )
    
    op.create_index(
        'idx_historical_analysis_created',
        'historical_analyses',
        ['created_at']
    )
    
    # Composite index for date range queries
    op.create_index(
        'idx_historical_analysis_date_range',
        'historical_analyses',
        ['norad_id', 'analysis_date']
    )
    
    # Maneuver table indexes
    op.create_index(
        'idx_maneuver_time',
        'maneuvers',
        ['maneuver_time']
    )
    
    op.create_index(
        'idx_maneuver_status',
        'maneuvers',
        ['status']
    )


def downgrade():
    """
    Downgrade database schema by removing added indexes.
    """
    # Remove spacecraft indexes
    op.drop_index('idx_spacecraft_norad_id', table_name='spacecraft')
    op.drop_index('idx_spacecraft_type', table_name='spacecraft')
    op.drop_index('idx_spacecraft_country', table_name='spacecraft')
    op.drop_index('idx_spacecraft_operational', table_name='spacecraft')
    
    # Remove conjunction indexes
    op.drop_index('idx_conjunction_secondary_object_id', table_name='conjunctions')
    op.drop_index('idx_conjunction_time', table_name='conjunctions')
    op.drop_index('idx_conjunction_probability', table_name='conjunctions')
    op.drop_index('idx_conjunction_status', table_name='conjunctions')
    op.drop_index('idx_conjunction_spacecraft_time', table_name='conjunctions')
    op.drop_index('idx_high_risk_conjunctions', table_name='conjunctions')
    
    # Remove historical analysis indexes
    op.drop_index('idx_historical_analysis_norad_id', table_name='historical_analyses')
    op.drop_index('idx_historical_analysis_date', table_name='historical_analyses')
    op.drop_index('idx_historical_analysis_threat', table_name='historical_analyses')
    op.drop_index('idx_historical_analysis_type', table_name='historical_analyses')
    op.drop_index('idx_historical_analysis_created', table_name='historical_analyses')
    op.drop_index('idx_historical_analysis_date_range', table_name='historical_analyses')
    
    # Remove maneuver indexes
    op.drop_index('idx_maneuver_time', table_name='maneuvers')
    op.drop_index('idx_maneuver_status', table_name='maneuvers') 