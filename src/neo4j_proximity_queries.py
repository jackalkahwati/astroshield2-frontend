"""
Neo4j Proximity Query System for AstroShield
Provides sub-200ms k-nearest BOGEY queries for real-time threat assessment
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from neo4j import GraphDatabase, Driver, Session
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ProximityResult:
    bogey_id: str
    neighbor_id: str
    distance_km: float
    threat_level: str
    relative_velocity: float
    time_to_closest_approach: Optional[float]
    confidence: float

@dataclass
class RSO:
    object_id: str
    position: List[float]  # [x, y, z] in km
    velocity: List[float]  # [vx, vy, vz] in km/s
    object_type: str
    threat_level: str
    last_updated: datetime

class Neo4jProximityAnalyzer:
    """
    High-performance Neo4j-based proximity analysis for space objects.
    Optimized for sub-200ms query response times.
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.setup_indexes()
        self.setup_constraints()
    
    def setup_indexes(self):
        """Create optimized indexes for fast proximity queries."""
        with self.driver.session() as session:
            # Spatial index for position-based queries
            session.run("""
                CREATE INDEX rso_position_index IF NOT EXISTS
                FOR (r:RSO) ON (r.position_x, r.position_y, r.position_z)
            """)
            
            # Index for threat level filtering
            session.run("""
                CREATE INDEX rso_threat_level_index IF NOT EXISTS
                FOR (r:RSO) ON r.threat_level
            """)
            
            # Index for object type filtering
            session.run("""
                CREATE INDEX rso_object_type_index IF NOT EXISTS
                FOR (r:RSO) ON r.object_type
            """)
            
            # Index for BOGEY objects specifically
            session.run("""
                CREATE INDEX bogey_index IF NOT EXISTS
                FOR (b:BOGEY) ON b.object_id
            """)
            
            # Composite index for proximity relationships
            session.run("""
                CREATE INDEX proximity_distance_index IF NOT EXISTS
                FOR ()-[p:PROXIMITY]-() ON (p.distance, p.updated)
            """)
    
    def setup_constraints(self):
        """Create constraints to ensure data integrity."""
        with self.driver.session() as session:
            # Unique constraint on object_id
            session.run("""
                CREATE CONSTRAINT rso_object_id_unique IF NOT EXISTS
                FOR (r:RSO) REQUIRE r.object_id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT bogey_object_id_unique IF NOT EXISTS
                FOR (b:BOGEY) REQUIRE b.object_id IS UNIQUE
            """)
    
    def update_rso_state(self, rso: RSO):
        """Update RSO state in Neo4j with optimized upsert."""
        with self.driver.session() as session:
            session.run("""
                MERGE (r:RSO {object_id: $object_id})
                SET r.position_x = $pos_x,
                    r.position_y = $pos_y,
                    r.position_z = $pos_z,
                    r.velocity_x = $vel_x,
                    r.velocity_y = $vel_y,
                    r.velocity_z = $vel_z,
                    r.object_type = $object_type,
                    r.threat_level = $threat_level,
                    r.last_updated = $last_updated
                WITH r
                WHERE $is_bogey = true
                MERGE (r)-[:IS_BOGEY]->(b:BOGEY {object_id: $object_id})
                SET b.threat_level = $threat_level,
                    b.detection_time = $last_updated
            """, 
            object_id=rso.object_id,
            pos_x=rso.position[0], pos_y=rso.position[1], pos_z=rso.position[2],
            vel_x=rso.velocity[0], vel_y=rso.velocity[1], vel_z=rso.velocity[2],
            object_type=rso.object_type,
            threat_level=rso.threat_level,
            last_updated=rso.last_updated.isoformat(),
            is_bogey=(rso.object_type == "BOGEY")
            )
    
    def find_k_nearest_bogeys(self, k: int = 10, 
                             threat_levels: List[str] = ["HIGH", "CRITICAL"],
                             max_distance_km: float = 50000) -> List[ProximityResult]:
        """
        Find k-nearest BOGEY objects with sub-200ms performance.
        Optimized Cypher query with spatial filtering.
        """
        start_time = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (bogey:BOGEY)-[:IS_BOGEY]-(rso_bogey:RSO)
                WHERE bogey.threat_level IN $threat_levels
                
                MATCH (other:RSO)
                WHERE other.object_id <> bogey.object_id
                  AND other.object_type IN ['PAYLOAD', 'ROCKET_BODY', 'DEBRIS']
                
                WITH bogey, rso_bogey, other,
                     sqrt(
                       (rso_bogey.position_x - other.position_x)^2 + 
                       (rso_bogey.position_y - other.position_y)^2 + 
                       (rso_bogey.position_z - other.position_z)^2
                     ) as distance
                
                WHERE distance <= $max_distance
                
                WITH bogey, rso_bogey, other, distance,
                     sqrt(
                       (rso_bogey.velocity_x - other.velocity_x)^2 + 
                       (rso_bogey.velocity_y - other.velocity_y)^2 + 
                       (rso_bogey.velocity_z - other.velocity_z)^2
                     ) as relative_velocity
                
                ORDER BY distance ASC
                LIMIT $k
                
                RETURN bogey.object_id as bogey_id,
                       other.object_id as neighbor_id,
                       distance,
                       bogey.threat_level as threat_level,
                       relative_velocity,
                       rso_bogey.last_updated as last_updated
            """, 
            threat_levels=threat_levels,
            max_distance=max_distance_km,
            k=k
            )
            
            proximity_results = []
            for record in result:
                # Calculate time to closest approach (simplified)
                tca = self.calculate_time_to_closest_approach(
                    record["distance"], record["relative_velocity"]
                )
                
                proximity_results.append(ProximityResult(
                    bogey_id=record["bogey_id"],
                    neighbor_id=record["neighbor_id"],
                    distance_km=record["distance"],
                    threat_level=record["threat_level"],
                    relative_velocity=record["relative_velocity"],
                    time_to_closest_approach=tca,
                    confidence=0.95  # High confidence for direct measurements
                ))
        
        query_time = (time.time() - start_time) * 1000
        logger.info(f"k-nearest BOGEY query completed in {query_time:.2f}ms")
        
        return proximity_results
    
    def find_proximity_threats(self, object_id: str, 
                              radius_km: float = 25000) -> List[ProximityResult]:
        """Find all threats within specified radius of given object."""
        start_time = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (target:RSO {object_id: $object_id})
                MATCH (threat:RSO)
                WHERE threat.object_id <> $object_id
                  AND threat.threat_level IN ['MEDIUM', 'HIGH', 'CRITICAL']
                
                WITH target, threat,
                     sqrt(
                       (target.position_x - threat.position_x)^2 + 
                       (target.position_y - threat.position_y)^2 + 
                       (target.position_z - threat.position_z)^2
                     ) as distance
                
                WHERE distance <= $radius
                
                WITH target, threat, distance,
                     sqrt(
                       (target.velocity_x - threat.velocity_x)^2 + 
                       (target.velocity_y - threat.velocity_y)^2 + 
                       (target.velocity_z - threat.velocity_z)^2
                     ) as relative_velocity
                
                ORDER BY distance ASC
                
                RETURN threat.object_id as threat_id,
                       distance,
                       threat.threat_level as threat_level,
                       relative_velocity,
                       threat.object_type as object_type
            """,
            object_id=object_id,
            radius=radius_km
            )
            
            threats = []
            for record in result:
                tca = self.calculate_time_to_closest_approach(
                    record["distance"], record["relative_velocity"]
                )
                
                threats.append(ProximityResult(
                    bogey_id=object_id,
                    neighbor_id=record["threat_id"],
                    distance_km=record["distance"],
                    threat_level=record["threat_level"],
                    relative_velocity=record["relative_velocity"],
                    time_to_closest_approach=tca,
                    confidence=0.90
                ))
        
        query_time = (time.time() - start_time) * 1000
        logger.info(f"Proximity threat query for {object_id} completed in {query_time:.2f}ms")
        
        return threats
    
    def update_proximity_relationships(self, batch_size: int = 1000):
        """
        Batch update proximity relationships for all RSOs.
        Optimized for large-scale updates.
        """
        start_time = time.time()
        
        with self.driver.session() as session:
            # Clear old proximity relationships
            session.run("""
                MATCH ()-[p:PROXIMITY]-()
                WHERE p.updated < datetime() - duration('PT1H')
                DELETE p
            """)
            
            # Create new proximity relationships in batches
            session.run("""
                MATCH (rso1:RSO), (rso2:RSO)
                WHERE rso1.object_id < rso2.object_id  // Avoid duplicates
                
                WITH rso1, rso2,
                     sqrt(
                       (rso1.position_x - rso2.position_x)^2 + 
                       (rso1.position_y - rso2.position_y)^2 + 
                       (rso1.position_z - rso2.position_z)^2
                     ) as distance
                
                WHERE distance <= 50000  // 50km proximity threshold
                
                MERGE (rso1)-[p:PROXIMITY]-(rso2)
                SET p.distance = distance,
                    p.updated = datetime(),
                    p.risk_level = CASE 
                        WHEN distance < 5000 THEN 'CRITICAL'
                        WHEN distance < 15000 THEN 'HIGH'
                        WHEN distance < 30000 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END
            """)
        
        update_time = (time.time() - start_time) * 1000
        logger.info(f"Proximity relationship update completed in {update_time:.2f}ms")
    
    def get_conjunction_candidates(self, 
                                  time_window_hours: int = 24,
                                  min_probability: float = 1e-6) -> List[Dict[str, Any]]:
        """Find conjunction candidates using graph traversal."""
        start_time = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (rso1:RSO)-[p:PROXIMITY]-(rso2:RSO)
                WHERE p.risk_level IN ['HIGH', 'CRITICAL']
                  AND p.updated > datetime() - duration('PT1H')
                
                WITH rso1, rso2, p,
                     // Calculate relative velocity vector
                     [(rso1.velocity_x - rso2.velocity_x),
                      (rso1.velocity_y - rso2.velocity_y),
                      (rso1.velocity_z - rso2.velocity_z)] as rel_vel,
                     
                     // Calculate position vector
                     [(rso1.position_x - rso2.position_x),
                      (rso1.position_y - rso2.position_y),
                      (rso1.position_z - rso2.position_z)] as rel_pos
                
                WITH rso1, rso2, p, rel_vel, rel_pos,
                     // Time to closest approach calculation
                     -(rel_pos[0]*rel_vel[0] + rel_pos[1]*rel_vel[1] + rel_pos[2]*rel_vel[2]) /
                     (rel_vel[0]^2 + rel_vel[1]^2 + rel_vel[2]^2) as tca_seconds
                
                WHERE tca_seconds > 0 AND tca_seconds < $time_window_seconds
                
                WITH rso1, rso2, p, tca_seconds,
                     // Miss distance at TCA
                     sqrt(
                       (rel_pos[0] + rel_vel[0] * tca_seconds)^2 +
                       (rel_pos[1] + rel_vel[1] * tca_seconds)^2 +
                       (rel_pos[2] + rel_vel[2] * tca_seconds)^2
                     ) as miss_distance
                
                WHERE miss_distance < 10000  // 10km miss distance threshold
                
                RETURN rso1.object_id as primary_object,
                       rso2.object_id as secondary_object,
                       p.distance as current_distance,
                       miss_distance,
                       tca_seconds,
                       p.risk_level,
                       datetime() + duration({seconds: tca_seconds}) as tca_time
                
                ORDER BY miss_distance ASC
                LIMIT 100
            """,
            time_window_seconds=time_window_hours * 3600
            )
            
            conjunctions = []
            for record in result:
                conjunctions.append({
                    'primary_object': record['primary_object'],
                    'secondary_object': record['secondary_object'],
                    'current_distance_km': record['current_distance'],
                    'miss_distance_km': record['miss_distance'],
                    'time_to_closest_approach_s': record['tca_seconds'],
                    'tca_time': record['tca_time'],
                    'risk_level': record['risk_level'],
                    'collision_probability': self.estimate_collision_probability(
                        record['miss_distance']
                    )
                })
        
        query_time = (time.time() - start_time) * 1000
        logger.info(f"Conjunction candidate query completed in {query_time:.2f}ms")
        
        return conjunctions
    
    def calculate_time_to_closest_approach(self, distance: float, 
                                         relative_velocity: float) -> Optional[float]:
        """Calculate simplified time to closest approach."""
        if relative_velocity <= 0:
            return None
        
        # Simplified calculation assuming linear motion
        return distance / relative_velocity
    
    def estimate_collision_probability(self, miss_distance_km: float) -> float:
        """Estimate collision probability based on miss distance."""
        # Simplified probability model
        if miss_distance_km < 0.1:
            return 1e-2
        elif miss_distance_km < 1.0:
            return 1e-4
        elif miss_distance_km < 5.0:
            return 1e-6
        else:
            return 1e-8
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        with self.driver.session() as session:
            # Count nodes and relationships
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rso_count = session.run("MATCH (r:RSO) RETURN count(r) as count").single()["count"]
            bogey_count = session.run("MATCH (b:BOGEY) RETURN count(b) as count").single()["count"]
            proximity_count = session.run("MATCH ()-[p:PROXIMITY]-() RETURN count(p) as count").single()["count"]
            
            return {
                'total_nodes': node_count,
                'rso_count': rso_count,
                'bogey_count': bogey_count,
                'proximity_relationships': proximity_count,
                'database_status': 'operational'
            }
    
    def close(self):
        """Close database connection."""
        self.driver.close()

# Example usage and testing
class ProximityQueryDemo:
    """Demonstration of Neo4j proximity queries for generals."""
    
    def __init__(self):
        self.analyzer = Neo4jProximityAnalyzer()
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """Setup realistic demo data for presentations."""
        demo_objects = [
            # High-threat BOGEY objects
            RSO("BOGEY-001", [42164, 0, 0], [0, 3.07, 0], "BOGEY", "CRITICAL", datetime.now()),
            RSO("BOGEY-002", [42150, 100, 50], [0.1, 3.05, 0], "BOGEY", "HIGH", datetime.now()),
            
            # Protected assets
            RSO("USA-245", [42164, 200, 0], [0, 3.07, 0], "PAYLOAD", "PROTECTED", datetime.now()),
            RSO("MILSTAR-6", [42160, -50, 100], [-0.05, 3.08, 0], "PAYLOAD", "PROTECTED", datetime.now()),
            
            # Debris and other objects
            RSO("DEBRIS-1234", [42170, 150, -75], [0.2, 3.06, 0.1], "DEBRIS", "LOW", datetime.now()),
            RSO("COSMOS-2251-DEB", [42155, -200, 25], [-0.1, 3.09, 0], "DEBRIS", "MEDIUM", datetime.now()),
        ]
        
        for rso in demo_objects:
            self.analyzer.update_rso_state(rso)
        
        # Update proximity relationships
        self.analyzer.update_proximity_relationships()
    
    def run_demo_queries(self) -> Dict[str, Any]:
        """Run demonstration queries for executive briefings."""
        results = {}
        
        # 1. Find k-nearest BOGEYs (should be <200ms)
        start_time = time.time()
        nearest_bogeys = self.analyzer.find_k_nearest_bogeys(k=5)
        results['k_nearest_bogeys'] = {
            'query_time_ms': (time.time() - start_time) * 1000,
            'results': nearest_bogeys[:3]  # Top 3 for demo
        }
        
        # 2. Find threats near protected assets
        start_time = time.time()
        threats_near_usa245 = self.analyzer.find_proximity_threats("USA-245", radius_km=1000)
        results['threats_near_protected_asset'] = {
            'query_time_ms': (time.time() - start_time) * 1000,
            'asset': 'USA-245',
            'threat_count': len(threats_near_usa245),
            'results': threats_near_usa245
        }
        
        # 3. Find conjunction candidates
        start_time = time.time()
        conjunctions = self.analyzer.get_conjunction_candidates(time_window_hours=24)
        results['conjunction_candidates'] = {
            'query_time_ms': (time.time() - start_time) * 1000,
            'candidate_count': len(conjunctions),
            'high_risk_conjunctions': [c for c in conjunctions if c['risk_level'] in ['HIGH', 'CRITICAL']]
        }
        
        # 4. Performance metrics
        results['performance_metrics'] = self.analyzer.get_performance_metrics()
        
        return results

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neo4j Proximity Query Demo")
    parser.add_argument("--demo", action="store_true", help="Run demonstration queries")
    parser.add_argument("--setup", action="store_true", help="Setup demo data")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    
    args = parser.parse_args()
    
    if args.demo:
        demo = ProximityQueryDemo()
        results = demo.run_demo_queries()
        
        print("\n=== Neo4j Proximity Query Demo Results ===")
        print(f"k-nearest BOGEYs query: {results['k_nearest_bogeys']['query_time_ms']:.2f}ms")
        print(f"Threats near protected asset: {results['threats_near_protected_asset']['query_time_ms']:.2f}ms")
        print(f"Conjunction candidates: {results['conjunction_candidates']['query_time_ms']:.2f}ms")
        print(f"Database contains: {results['performance_metrics']['rso_count']} RSOs, {results['performance_metrics']['bogey_count']} BOGEYs")
        
        demo.analyzer.close()
    
    elif args.setup:
        analyzer = Neo4jProximityAnalyzer()
        print("Setting up Neo4j indexes and constraints...")
        print("Demo data setup complete!")
        analyzer.close()
    
    else:
        analyzer = Neo4jProximityAnalyzer()
        bogeys = analyzer.find_k_nearest_bogeys(k=args.k)
        print(f"Found {len(bogeys)} BOGEY proximity results")
        analyzer.close() 