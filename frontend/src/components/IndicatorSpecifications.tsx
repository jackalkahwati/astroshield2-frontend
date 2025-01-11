import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Grid,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';

interface BaseIndicatorSpec {
  description: string;
  pass_criteria: string;
  features: string[];
  type: 'ml' | 'rule' | 'threshold';
}

interface MLIndicatorSpec extends BaseIndicatorSpec {
  type: 'ml';
  algorithm: string;
  confidence_threshold: number;
  model_version: string;
  last_trained: string;
  confidence_score: number;
  training_accuracy: number;
}

interface RuleIndicatorSpec extends BaseIndicatorSpec {
  type: 'rule';
  rule_set: string;
  rule_version: string;
  last_updated: string;
}

interface ThresholdIndicatorSpec extends BaseIndicatorSpec {
  type: 'threshold';
  threshold_value: number;
  unit: string;
  comparison: 'greater' | 'less' | 'equal';
}

type IndicatorSpec = MLIndicatorSpec | RuleIndicatorSpec | ThresholdIndicatorSpec;

interface CategorySpecs {
  [key: string]: IndicatorSpec;
}

interface AllSpecs {
  [category: string]: CategorySpecs;
}

const mockSpecs: AllSpecs = {
  stability: {
    object_stability: {
      type: 'ml',
      algorithm: 'LSTM Neural Network',
      description: 'Evaluates if the object is maintaining stable orbit and behavior',
      pass_criteria: 'No significant deviations from expected orbital parameters',
      confidence_threshold: 0.95,
      features: ['Orbital Elements', 'Historical Stability', 'Attitude Data'],
      model_version: 'v2.3.1',
      last_trained: '2024-03-15',
      confidence_score: 0.97,
      training_accuracy: 0.96
    },
    stability_change: {
      type: 'ml',
      algorithm: 'Change Point Detection',
      description: 'Detects changes in stability patterns over time',
      pass_criteria: 'No unexplained stability changes',
      confidence_threshold: 0.92,
      features: ['Stability Metrics', 'Environmental Data'],
      model_version: 'v2.1.1',
      last_trained: '2024-03-10',
      confidence_score: 0.95,
      training_accuracy: 0.94
    }
  },
  maneuvers: {
    maneuvers_detected: {
      type: 'ml',
      algorithm: 'Bi-LSTM with Attention',
      description: 'Identifies and classifies orbital maneuvers',
      pass_criteria: 'All maneuvers match declared operations',
      confidence_threshold: 0.92,
      features: ['Trajectory Data', 'Historical Maneuvers'],
      model_version: 'v3.0.1',
      last_trained: '2024-03-18',
      confidence_score: 0.96,
      training_accuracy: 0.95
    },
    imaging_maneuvers: {
      type: 'ml',
      algorithm: 'Pattern Recognition CNN',
      description: 'Detects maneuvers resulting in valid remote-sensing passes',
      pass_criteria: 'Maneuver patterns consistent with imaging operations',
      confidence_threshold: 0.90,
      features: ['Maneuver History', 'Ground Track Analysis'],
      model_version: 'v2.5.0',
      last_trained: '2024-03-16',
      confidence_score: 0.94,
      training_accuracy: 0.93
    },
    coverage_gap_maneuvers: {
      type: 'ml',
      algorithm: 'Temporal Pattern Mining',
      description: 'Identifies maneuvers occurring in sensor coverage gaps',
      pass_criteria: 'No unexplained maneuvers during gaps',
      confidence_threshold: 0.88,
      features: ['Coverage Data', 'Maneuver History'],
      model_version: 'v2.1.0',
      last_trained: '2024-03-13',
      confidence_score: 0.92,
      training_accuracy: 0.91
    }
  },
  rf_indicators: {
    rf_detected: {
      type: 'ml',
      algorithm: 'Deep Neural Network',
      description: 'Detects and characterizes RF emissions',
      pass_criteria: 'RF emissions match declared capabilities',
      confidence_threshold: 0.93,
      features: ['RF Spectrum', 'Signal Characteristics'],
      model_version: 'v2.2.0',
      last_trained: '2024-03-14',
      confidence_score: 0.95,
      training_accuracy: 0.94
    },
    rf_pol_violation: {
      type: 'ml',
      algorithm: 'Pattern Analysis',
      description: 'Identifies RF pattern-of-life violations',
      pass_criteria: 'RF activity within normal parameters',
      confidence_threshold: 0.91,
      features: ['Historical RF Data', 'Activity Patterns'],
      model_version: 'v2.0.1',
      last_trained: '2024-03-11',
      confidence_score: 0.93,
      training_accuracy: 0.92
    },
    itu_fcc_violation: {
      type: 'rule',
      description: 'Checks compliance with ITU and FCC filings',
      pass_criteria: 'All RF activities comply with filings',
      rule_set: 'ITU/FCC Compliance Rules',
      rule_version: 'v1.9.0',
      last_updated: '2024-03-13',
      features: ['RF Parameters', 'Filing Data']
    }
  },
  physical_characteristics: {
    subsatellite_deployment: {
      type: 'ml',
      algorithm: 'Object Detection',
      description: 'Detects deployment of sub-satellites',
      pass_criteria: 'All deployments match declarations',
      confidence_threshold: 0.94,
      features: ['Tracking Data', 'Object Count'],
      model_version: 'v1.7.1',
      last_trained: '2024-03-09',
      confidence_score: 0.96,
      training_accuracy: 0.95
    },
    amr_anomaly: {
      type: 'ml',
      algorithm: 'Physics-based ML',
      description: 'Detects anomalies in area-to-mass ratio',
      pass_criteria: 'AMR within expected range',
      confidence_threshold: 0.92,
      features: ['AMR Data', 'Object Properties'],
      model_version: 'v2.4.0',
      last_trained: '2024-03-17',
      confidence_score: 0.95,
      training_accuracy: 0.94
    },
    signature_mismatch: {
      type: 'ml',
      algorithm: 'Multi-modal Fusion',
      description: 'Detects mismatches between optical and RADAR signatures',
      pass_criteria: 'Signatures match expected characteristics',
      confidence_threshold: 0.90,
      features: ['Optical Data', 'RADAR Data'],
      model_version: 'v2.1.2',
      last_trained: '2024-03-14',
      confidence_score: 0.93,
      training_accuracy: 0.92
    }
  },
  orbital_characteristics: {
    orbit_family: {
      type: 'ml',
      algorithm: 'Clustering Analysis',
      description: 'Identifies if orbit is out of family',
      pass_criteria: 'Orbit matches expected family',
      confidence_threshold: 0.89,
      features: ['Orbital Elements', 'Historical Data'],
      model_version: 'v1.5.0',
      last_trained: '2024-03-08',
      confidence_score: 0.91,
      training_accuracy: 0.90
    },
    high_radiation_orbit: {
      type: 'threshold',
      description: 'Monitors if object is in high radiation environment',
      pass_criteria: 'Radiation levels within acceptable range',
      threshold_value: 1000,
      unit: 'rad/day',
      comparison: 'less',
      features: ['Orbit Parameters', 'Space Weather']
    },
    unoccupied_orbit: {
      type: 'rule',
      description: 'Checks if object is in relatively unoccupied orbit',
      pass_criteria: 'Object density below threshold',
      rule_set: 'Orbit Population Rules',
      rule_version: 'v1.1.0',
      last_updated: '2024-03-04',
      features: ['Space Object Catalog', 'Orbit Statistics']
    }
  },
  launch_indicators: {
    threat_launch_origin: {
      type: 'rule',
      description: 'Identifies if object came from known threat launch site',
      pass_criteria: 'Launch site not associated with threats',
      rule_set: 'Launch Site Classification',
      rule_version: 'v1.2.1',
      last_updated: '2024-03-05',
      features: ['Launch Site Data', 'Historical Launches']
    },
    unexpected_objects: {
      type: 'ml',
      algorithm: 'Object Tracking',
      description: 'Detects if number of tracked objects exceeds expected count',
      pass_criteria: 'Object count matches launch manifest',
      confidence_threshold: 0.95,
      features: ['Launch Data', 'Object Tracking'],
      model_version: 'v2.3.1',
      last_trained: '2024-03-15',
      confidence_score: 0.97,
      training_accuracy: 0.96
    }
  },
  compliance: {
    un_registry: {
      type: 'rule',
      description: 'Checks if object is in UN satellite registry',
      pass_criteria: 'Object properly registered',
      rule_set: 'UN Registry Compliance',
      rule_version: 'v2.1.0',
      last_updated: '2024-03-20',
      features: ['UN Registry Data', 'Object Identity']
    },
    analyst_disagreement: {
      type: 'ml',
      algorithm: 'Consensus Analysis',
      description: 'Identifies disagreements between analysts',
      pass_criteria: 'No significant analyst disagreements',
      confidence_threshold: 0.90,
      features: ['Analyst Reports', 'Historical Assessments'],
      model_version: 'v1.5.2',
      last_trained: '2024-03-18',
      confidence_score: 0.94,
      training_accuracy: 0.93
    }
  }
};

const IndicatorSpecifications: React.FC = () => {
  const renderIndicatorContent = (name: string, spec: IndicatorSpec) => {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ textTransform: 'capitalize' }}>
              {name.replace('_', ' ')}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {spec.type === 'ml' && (
                <>
                  <Chip 
                    label={`ML ${spec.model_version}`} 
                    size="small" 
                    color="primary"
                  />
                  <Tooltip title={`Last trained: ${spec.last_trained}`}>
                    <InfoIcon color="action" fontSize="small" />
                  </Tooltip>
                </>
              )}
              {spec.type === 'rule' && (
                <>
                  <Chip 
                    label={`Rule ${spec.rule_version}`} 
                    size="small" 
                    color="secondary"
                  />
                  <Tooltip title={`Last updated: ${spec.last_updated}`}>
                    <InfoIcon color="action" fontSize="small" />
                  </Tooltip>
                </>
              )}
              {spec.type === 'threshold' && (
                <Chip 
                  label={`Threshold: ${spec.threshold_value}${spec.unit}`} 
                  size="small" 
                  color="info"
                />
              )}
            </Box>
          </Box>

          <Typography variant="body2" color="text.secondary" paragraph>
            {spec.description}
          </Typography>

          {spec.type === 'ml' && (
            <>
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">Confidence Score</Typography>
                  <Typography variant="body2" color="primary">
                    {(spec.confidence_score * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={spec.confidence_score * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">Training Accuracy</Typography>
                  <Typography variant="body2" color="success.main">
                    {(spec.training_accuracy * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={spec.training_accuracy * 100}
                  sx={{ 
                    height: 8, 
                    borderRadius: 4,
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: 'success.main'
                    }
                  }}
                />
              </Box>

              <Typography variant="body2" gutterBottom>
                <strong>Algorithm:</strong> {spec.algorithm}
              </Typography>
            </>
          )}

          {spec.type === 'rule' && (
            <Typography variant="body2" gutterBottom>
              <strong>Rule Set:</strong> {spec.rule_set}
            </Typography>
          )}

          {spec.type === 'threshold' && (
            <Typography variant="body2" gutterBottom>
              <strong>Threshold:</strong> {spec.comparison === 'greater' ? '>' : spec.comparison === 'less' ? '<' : '='} {spec.threshold_value}{spec.unit}
            </Typography>
          )}
          
          <Typography variant="body2" gutterBottom>
            <strong>Pass Criteria:</strong> {spec.pass_criteria}
          </Typography>

          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              <strong>Features:</strong>
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {spec.features.map((feature) => (
                <Chip
                  key={feature}
                  label={feature}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box>
      {Object.entries(mockSpecs).map(([category, specs]) => (
        <Accordion key={category} defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
              {category.replace('_', ' ')}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              {Object.entries(specs).map(([name, spec]) => (
                <Grid item xs={12} md={6} key={name}>
                  {renderIndicatorContent(name, spec)}
                </Grid>
              ))}
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default IndicatorSpecifications; 