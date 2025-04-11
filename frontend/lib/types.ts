// Types for the AstroShield application

// API Response Types
export interface ApiResponse<T = any> {
  data: T | null;
  status?: number;
  error?: ApiError;
}

export interface ApiError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}

export interface RateLimitError extends ApiError {
  retryAfter: number;
}

export interface NetworkError extends ApiError {
  type: 'network';
}

export interface ValidationError extends ApiError {
  validationErrors: Record<string, string>;
}

export interface EnhancedApiError extends ApiError {
  type: string;
  statusCode?: number;
  timestamp: string;
  path: string;
}

// Comprehensive Data
export interface ComprehensiveData {
  status: string;
  timestamp: string;
  metrics: Record<string, number>;
  alerts: string[];
}

// System Health
export interface SystemHealth {
  status: string;
  version: string;
  uptime: number;
  memory: {
    total: number;
    used: number;
    free: number;
  };
  services: Record<string, {
    status: string;
    lastCheck: string;
  }>;
}

// Satellite Data
export interface SatelliteData {
  id: string;
  name: string;
  norad_id?: string;
  owner: string;
  status: string;
  orbit_type: string;
  launch_date: string;
  position?: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  metadata?: Record<string, any>;
}

// Telemetry
export interface TelemetryData {
  timestamp: string;
  data_points: Record<string, number>;
}

// Security and Monitoring
export interface SecurityHeaders {
  'Content-Security-Policy': string;
  'Strict-Transport-Security': string;
  'X-Content-Type-Options': string;
  'X-Frame-Options': string;
  'X-XSS-Protection': string;
}

export interface MonitoringMetrics {
  requestLatency: number[];
  errorRate: number;
  activeConnections: number;
  requestCount: number;
}

// Welder's Arc event types
export enum EventType {
  LAUNCH = "launch",
  REENTRY = "reentry",
  MANEUVER = "maneuver",
  SEPARATION = "separation",
  PROXIMITY = "proximity",
  LINK_CHANGE = "link_change",
  ATTITUDE_CHANGE = "attitude_change"
}

// Event status
export enum EventStatus {
  DETECTED = "detected",
  PROCESSING = "processing",
  AWAITING_DATA = "awaiting_data",
  COMPLETED = "completed",
  REJECTED = "rejected",
  ERROR = "error"
}

// Threat levels
export enum ThreatLevel {
  NONE = "none",
  LOW = "low",
  MODERATE = "moderate",
  HIGH = "high",
  SEVERE = "severe"
}

// Course of action
export interface CourseOfAction {
  title: string;
  description: string;
  priority: number; // 1-5
  actions: string[];
  expiration?: string; // ISO date string
}

// Event processing step
export interface EventProcessingStep {
  step_name: string;
  timestamp: string; // ISO date string
  status: string;
  output?: Record<string, any>;
  error?: string;
}

// Event detection data
export interface EventDetection {
  event_type: EventType;
  object_id: string;
  detection_time: string; // ISO date string
  confidence: number; // 0.0-1.0
  sensor_data: Record<string, any>;
  metadata?: Record<string, any>;
}

// Event model
export interface Event {
  id: string;
  event_type: EventType;
  object_id: string;
  status: EventStatus;
  creation_time: string; // ISO date string
  update_time: string; // ISO date string
  detection_data: EventDetection;
  processing_steps: EventProcessingStep[];
  hostility_assessment?: Record<string, any>;
  threat_level?: ThreatLevel;
  coa_recommendation?: CourseOfAction;
}

// Event query parameters
export interface EventQuery {
  event_types?: EventType[];
  object_ids?: string[];
  status?: EventStatus[];
  start_time?: string; // ISO date string
  end_time?: string; // ISO date string
  limit?: number;
  offset?: number;
}

// Dashboard data
export interface EventDashboardData {
  total_events: number;
  events_by_type: Record<string, number>;
  events_by_status: Record<string, number>;
  events_by_threat: Record<string, number>;
  recent_high_threats: Event[];
}

// Maneuver data
export interface ManeuverDetails {
  delta_v?: number;
  duration?: number;
  fuel_required?: number;
  direction?: {
    x: number;
    y: number;
    z: number;
  };
  target_orbit?: {
    altitude: number;
    inclination: number;
    eccentricity: number;
  };
}

export interface ManeuverData {
  id: string;
  satellite_id: string;
  type: string;
  status: string;
  scheduledTime: string; // ISO date string
  completedTime?: string; // ISO date string
  details?: ManeuverDetails;
  created_by?: string;
  created_at: string; // ISO date string
  updated_at?: string; // ISO date string
}

// Specific event detection models (for reference)

export interface LaunchDetection {
  object_id: string;
  launch_site: { lat: number; lon: number };
  launch_time: string;
  initial_trajectory: number[];
  confidence: number;
  sensor_id: string;
}

export interface ReentryDetection {
  object_id: string;
  predicted_reentry_time: string;
  predicted_location: { lat: number; lon: number };
  confidence: number;
  sensor_id: string;
}

export interface ManeuverDetection {
  object_id: string;
  maneuver_time: string;
  delta_v: number;
  direction: number[];
  confidence: number;
  sensor_id: string;
}

export interface SeparationDetection {
  parent_object_id: string;
  child_object_id: string;
  separation_time: string;
  relative_velocity: number;
  confidence: number;
  sensor_id: string;
}

export interface ProximityDetection {
  primary_object_id: string;
  secondary_object_id: string;
  closest_approach_time: string;
  minimum_distance: number;
  relative_velocity: number;
  confidence: number;
  sensor_id: string;
}

export interface LinkChangeDetection {
  object_id: string;
  link_change_time: string;
  link_type: string;
  previous_state: string;
  current_state: string;
  confidence: number;
  sensor_id: string;
}

export interface AttitudeChangeDetection {
  object_id: string;
  attitude_change_time: string;
  change_magnitude: number;
  previous_attitude: number[];
  current_attitude: number[];
  confidence: number;
  sensor_id: string;
}

// Trajectory types
export interface TrajectoryObjectProperties {
  mass: number;
  area: number;
  cd: number;
}

export interface TrajectoryBreakupModel {
  enabled: boolean;
  fragment_count: number;
  mass_distribution?: string;
  velocity_perturbation?: number;
}

export interface TrajectoryConfig {
  atmospheric_model: string; // e.g. "nrlmsise", "exponential", "jacchia"
  wind_model: string; // e.g. "hwm14", "custom"
  monte_carlo_samples: number;
  object_properties: TrajectoryObjectProperties;
  breakup_model: TrajectoryBreakupModel;
}

export interface TrajectoryRequest {
  config: TrajectoryConfig;
  initial_state: number[]; // [x, y, z, vx, vy, vz]
}

export interface TrajectoryPoint {
  time: number;
  position: [number, number, number];
  velocity: [number, number, number];
}

export interface BreakupEvent {
  time: string;
  altitude: number;
  fragments: number;
}

export interface ImpactPrediction {
  time: string;
  location: {
    lat: number;
    lon: number;
  };
  velocity: {
    magnitude: number;
    direction: {
      x: number;
      y: number;
      z: number;
    };
  };
  uncertainty_radius_km: number;
  confidence: number;
  monte_carlo_stats?: {
    samples: number;
    time_std: number;
    position_std: number;
    velocity_std: number;
  };
}

export interface TrajectoryResult {
  trajectory: TrajectoryPoint[];
  impact_prediction: ImpactPrediction;
  breakup_events?: BreakupEvent[];
}