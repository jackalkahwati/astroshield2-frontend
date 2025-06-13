# Kafka Message Flow Monitoring Dashboard

## Overview

A comprehensive real-time monitoring dashboard for Kafka message flows in the AstroShield application. This dashboard provides complete visibility into SDA (Space Data Association) Kafka integration, message routing, performance metrics, and system health.

## Features

### 🔴 Live Message Stream
- **Real-time WebSocket connection** for live message viewing
- **Bi-directional monitoring** (incoming/outgoing messages)
- **Message content inspection** with detailed payload viewer
- **Search and filtering** by topic, subsystem, or message type
- **Pause/resume functionality** for debugging
- **Message statistics** and throughput tracking

### 📊 Topic Health Dashboard
- **SDA subsystem organization** (SS0-SS6 topics)
- **Health status indicators** (healthy, warning, error, offline)
- **Consumer lag monitoring** per topic
- **Message count tracking** and last message timestamps
- **Partition and replication factor** visibility
- **Color-coded status** by subsystem

### ⚡ Performance Metrics
- **Real-time throughput** (messages/second, bytes/second)
- **Latency distribution** (average, P95, P99)
- **Error rate monitoring** with recent error details
- **Consumer lag analysis** across all topics
- **Trend indicators** showing performance changes
- **Health thresholds** with automatic status determination

### 🔧 Subscription Manager
- **Active subscription monitoring** for all consumer groups
- **Subscription controls** (start/stop/restart)
- **Lag monitoring** per subscription
- **Message processing statistics**
- **Error tracking** per subscription
- **Heartbeat monitoring** for consumer health

### 🌐 Message Flow Visualization
- **Visual message routing** from producers to consumers
- **SDA subsystem integration** showing data flow
- **Connection status** between components
- **Message volume indicators** for each connection
- **Interactive flow diagram** with status indicators

## Architecture

### Frontend Components

```
components/monitoring/
├── LiveMessageStream.tsx          # Real-time message viewer
├── TopicHealthDashboard.tsx       # Topic status monitoring
├── PerformanceMetrics.tsx         # Performance KPIs
├── SubscriptionManager.tsx        # Consumer management
├── MessageFlowVisualization.tsx   # Flow diagram
└── README.md                      # This file
```

### Backend API Endpoints

```
/api/v1/kafka-monitor/
├── /system-status                 # Overall system health
├── /live-stream (WebSocket)       # Real-time message stream
├── /topics/health                 # Topic health status
├── /metrics/performance           # Performance metrics
├── /subscriptions                 # Active subscriptions
├── /subscriptions/{id}/{action}   # Subscription controls
└── /message-flow                  # Flow visualization data
```

### TypeScript Interfaces

```typescript
// Core message interface
interface KafkaMessage {
  id: string
  topic: string
  direction: 'incoming' | 'outgoing'
  timestamp: string
  messageType: string
  subsystem: string
  sourceSystem: string
  content: any
  size: number
  status: 'success' | 'error' | 'pending'
  latency?: number
  correlationId?: string
}

// Topic health monitoring
interface TopicHealth {
  name: string
  status: 'healthy' | 'warning' | 'error' | 'offline'
  messageCount: number
  lastMessage: string | null
  consumerLag: number
  partitions: number
  replicationFactor: number
  category: 'sda' | 'internal' | 'test'
  subsystem?: string
}

// Performance metrics
interface KafkaMetrics {
  throughput: {
    messagesPerSecond: number
    bytesPerSecond: number
  }
  latency: {
    average: number
    p95: number
    p99: number
  }
  errors: {
    rate: number
    total: number
    recentErrors: Array<{
      timestamp: string
      topic: string
      error: string
    }>
  }
  consumers: {
    totalLag: number
    activeConsumers: number
  }
}
```

## SDA Integration

### Supported SDA Topics

The dashboard monitors all official SDA Welders Arc topics:

#### SS0 - Data Ingestion
- `ss0.data.launch-detection`
- `ss0.sensor.heartbeat`
- `ss0.data.weather.*` (9 weather-related topics)

#### SS2 - State Estimation
- `ss2.data.elset.sgp4`
- `ss2.data.elset.sgp4-xp`
- `ss2.data.state-vector.best-state`
- `ss2.data.observation-track`

#### SS4 - CCDM (Conjunction Detection)
- `ss4.maneuver.detection`
- `ss4.conjunction.warning`
- `ss4.ccdm.detection`

#### SS5 - Hostility Monitoring (19 official topics)
- `ss5.launch.*` (9 launch-related topics)
- `ss5.pez-wez-prediction.*` (5 weapon engagement zones)
- `ss5.reentry.prediction`
- `ss5.separation.detection`

#### SS6 - Threat Response
- `ss6.response-recommendation.launch`
- `ss6.response-recommendation.on-orbit`

### Schema Validation

The dashboard integrates with the SDA schema validation system:
- **Message format validation** against SDA schemas
- **Schema version tracking** and compatibility checks
- **Error reporting** for invalid messages
- **Automatic schema detection** from message headers

## Usage

### Accessing the Dashboard

1. Navigate to `/kafka-monitor` in the AstroShield application
2. The dashboard loads with real-time data automatically
3. Use the tabbed interface to switch between different views

### Live Monitoring

1. **Live Stream Tab**: Watch messages in real-time
   - Use search to filter by topic or content
   - Click messages to view detailed payload
   - Pause stream for detailed inspection

2. **Topic Health Tab**: Monitor topic status
   - View health by SDA subsystem
   - Check consumer lag and message counts
   - Identify problematic topics quickly

3. **Performance Tab**: Track system performance
   - Monitor throughput and latency trends
   - View error rates and recent failures
   - Check consumer lag distribution

4. **Subscriptions Tab**: Manage consumers
   - Start/stop individual subscriptions
   - Monitor processing rates and errors
   - View heartbeat status

5. **Flow Diagram Tab**: Visualize message routing
   - See data flow from producers to consumers
   - Identify bottlenecks in the pipeline
   - Monitor connection health

### Real-time Updates

- **WebSocket connection** provides live message streaming
- **Automatic refresh** every 5-30 seconds for metrics
- **Status indicators** show connection health
- **Graceful degradation** when Kafka is unavailable

## Technical Implementation

### WebSocket Integration

```typescript
// Real-time message streaming
useEffect(() => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/api/v1/kafka-monitor/live-stream`
  
  wsRef.current = new WebSocket(wsUrl)
  
  wsRef.current.onmessage = (event) => {
    const wsMessage: WebSocketMessage = JSON.parse(event.data)
    if (wsMessage.type === 'message' && !isPaused) {
      const newMessage: KafkaMessage = wsMessage.data
      setMessages(prev => [newMessage, ...prev.slice(0, 499)])
    }
  }
}, [isPaused])
```

### Enhanced Kafka Client

The dashboard integrates with an enhanced Kafka client that provides monitoring hooks:

```python
class MonitoredKafkaClient:
    def __init__(self):
        self.message_buffer = []
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "last_activity": datetime.now(timezone.utc)
        }
    
    async def publish_with_monitoring(self, topic: str, message: dict):
        # Add monitoring capabilities to publish operations
        monitored_message = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "direction": "outgoing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": message,
            "status": "success"
        }
        
        self.message_buffer.append(monitored_message)
        
        # Send to WebSocket clients
        await manager.send_message(json.dumps({
            "type": "message",
            "data": monitored_message
        }))
```

### Mock Data Support

For development and testing, the dashboard includes comprehensive mock data:
- **Realistic message generation** with proper SDA schemas
- **Configurable message rates** and error injection
- **Topic health simulation** with various status conditions
- **Performance metric simulation** with trends

## Configuration

### Environment Variables

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=SCRAM-SHA-512

# SDA Integration
SDA_KAFKA_BOOTSTRAP_SERVERS=kafka.sda.mil:9092
SDA_KAFKA_USERNAME=your_username
SDA_KAFKA_PASSWORD=your_password

# Monitoring Settings
KAFKA_MONITOR_REFRESH_INTERVAL=5000
KAFKA_MONITOR_MESSAGE_BUFFER_SIZE=500
KAFKA_MONITOR_WEBSOCKET_TIMEOUT=30000
```

### Customization

The dashboard is highly customizable:
- **Topic filtering** by category or subsystem
- **Metric thresholds** for health determination
- **Refresh intervals** for different data types
- **Color schemes** and status indicators
- **Message retention** policies

## Benefits

### Operational Visibility
- **Complete system transparency** for Kafka operations
- **Proactive issue detection** through health monitoring
- **Performance optimization** through metrics analysis
- **Debugging capabilities** with message inspection

### SDA Compliance
- **Full SDA topic coverage** across all subsystems
- **Schema validation** for message compliance
- **Official topic naming** convention support
- **Subsystem integration** visualization

### Development Support
- **Mock data generation** for testing
- **Message debugging** with payload inspection
- **Performance profiling** for optimization
- **Integration testing** support

### Production Readiness
- **High availability** with graceful degradation
- **Scalable architecture** for high throughput
- **Security integration** with SDA authentication
- **Monitoring alerts** for operational issues

## Future Enhancements

- **Alert system** with configurable thresholds
- **Historical data** analysis and trending
- **Advanced filtering** and search capabilities
- **Export functionality** for metrics and logs
- **Integration** with external monitoring systems
- **Custom dashboards** for specific use cases
- **Machine learning** for anomaly detection
- **Performance optimization** recommendations

This dashboard provides comprehensive monitoring capabilities for AstroShield's Kafka integration, enabling operators to maintain high system reliability and performance while ensuring full compliance with SDA requirements. 