"""
Node-RED Integration Service
Visual workflow orchestration for CCDM indicators
"""

import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeREDConfig(BaseModel):
    """Node-RED configuration"""
    base_url: str = Field(default="http://localhost:1880")
    username: Optional[str] = None
    password: Optional[str] = None
    flow_prefix: str = Field(default="welders-arc")


class CCDMIndicator(BaseModel):
    """CCDM indicator from Problem 16"""
    indicator_id: str
    name: str
    category: str
    status: str  # PASS, FAIL, WARNING, ANALYZING
    confidence: float
    timestamp: datetime
    data: Dict[str, Any]


class NodeREDFlow(BaseModel):
    """Node-RED flow definition"""
    id: str
    label: str
    nodes: List[Dict[str, Any]]
    wires: List[List[str]]


class NodeREDService:
    """Service for managing Node-RED workflows"""
    
    def __init__(self, config: NodeREDConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.flow_cache: Dict[str, NodeREDFlow] = {}
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        
    async def connect(self):
        """Initialize connection to Node-RED"""
        auth = None
        if self.config.username and self.config.password:
            auth = aiohttp.BasicAuth(self.config.username, self.config.password)
            
        self.session = aiohttp.ClientSession(auth=auth)
        
    async def disconnect(self):
        """Close Node-RED connection"""
        if self.session:
            await self.session.close()
            
    async def deploy_ccdm_workflow(self) -> str:
        """Deploy complete CCDM indicator workflow"""
        flow = self._create_ccdm_flow()
        
        # Deploy flow to Node-RED
        async with self.session.post(
            f"{self.config.base_url}/flow",
            json=flow.dict()
        ) as response:
            response.raise_for_status()
            data = await response.json()
            flow_id = data["id"]
            
        self.flow_cache[flow_id] = flow
        logger.info(f"Deployed CCDM workflow: {flow_id}")
        
        return flow_id
        
    def _create_ccdm_flow(self) -> NodeREDFlow:
        """Create CCDM indicator processing flow"""
        nodes = []
        node_id = 1
        
        # Input nodes for each indicator category
        stability_input = self._create_node(
            node_id, "kafka-consumer", 
            {"topic": "welders.ss4.stability.indicators", "name": "Stability Indicators"}
        )
        nodes.append(stability_input)
        node_id += 1
        
        maneuver_input = self._create_node(
            node_id, "kafka-consumer",
            {"topic": "welders.ss4.maneuver.indicators", "name": "Maneuver Indicators"}
        )
        nodes.append(maneuver_input)
        node_id += 1
        
        rf_input = self._create_node(
            node_id, "kafka-consumer",
            {"topic": "welders.ss4.rf.indicators", "name": "RF Indicators"}
        )
        nodes.append(rf_input)
        node_id += 1
        
        # Indicator processing nodes
        stability_processor = self._create_node(
            node_id, "function",
            {
                "name": "Process Stability",
                "func": """
                    const indicator = msg.payload;
                    
                    // Apply stability thresholds
                    if (indicator.stability_score < 0.3) {
                        indicator.status = 'FAIL';
                        indicator.alert = true;
                    } else if (indicator.stability_score < 0.7) {
                        indicator.status = 'WARNING';
                    } else {
                        indicator.status = 'PASS';
                    }
                    
                    msg.payload = indicator;
                    return msg;
                """
            }
        )
        nodes.append(stability_processor)
        node_id += 1
        
        # Aggregator node
        aggregator = self._create_node(
            node_id, "join",
            {
                "name": "Aggregate Indicators",
                "mode": "custom",
                "build": "object",
                "property": "payload",
                "propertyType": "msg",
                "key": "topic",
                "joiner": "\\n",
                "joinerType": "str",
                "accumulate": True,
                "timeout": "5",
                "count": "",
                "topics": []
            }
        )
        nodes.append(aggregator)
        node_id += 1
        
        # CCDM scoring node
        scorer = self._create_node(
            node_id, "function",
            {
                "name": "CCDM Scoring",
                "func": """
                    const indicators = msg.payload;
                    let totalScore = 0;
                    let weights = {
                        stability: 0.15,
                        maneuver: 0.20,
                        rf: 0.15,
                        compliance: 0.10,
                        signature: 0.15,
                        physical: 0.15,
                        deception: 0.10
                    };
                    
                    // Calculate weighted score
                    for (const [category, data] of Object.entries(indicators)) {
                        const weight = weights[category] || 0.1;
                        const score = data.confidence * (data.status === 'FAIL' ? 1 : 0);
                        totalScore += score * weight;
                    }
                    
                    msg.payload = {
                        object_id: msg.object_id,
                        ccdm_score: totalScore,
                        indicators: indicators,
                        timestamp: new Date().toISOString()
                    };
                    
                    return msg;
                """
            }
        )
        nodes.append(scorer)
        node_id += 1
        
        # Output to object interest list
        output = self._create_node(
            node_id, "kafka-producer",
            {
                "topic": "welders.ss4.object.interest.list",
                "name": "Publish to OIL"
            }
        )
        nodes.append(output)
        
        # Create wiring
        wires = [
            [stability_input["id"], stability_processor["id"]],
            [stability_processor["id"], aggregator["id"]],
            [maneuver_input["id"], aggregator["id"]],
            [rf_input["id"], aggregator["id"]],
            [aggregator["id"], scorer["id"]],
            [scorer["id"], output["id"]]
        ]
        
        return NodeREDFlow(
            id=f"{self.config.flow_prefix}-ccdm-main",
            label="CCDM Indicator Processing",
            nodes=nodes,
            wires=wires
        )
        
    def _create_node(self, node_id: int, node_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Node-RED node definition"""
        return {
            "id": f"node-{node_id}",
            "type": node_type,
            "x": 100 + (node_id % 5) * 200,
            "y": 100 + (node_id // 5) * 100,
            **config
        }
        
    async def create_indicator_workflow(self, indicator_name: str, model_type: str) -> str:
        """Create workflow for specific CCDM indicator"""
        flow = NodeREDFlow(
            id=f"{self.config.flow_prefix}-{indicator_name}",
            label=f"CCDM {indicator_name} Indicator",
            nodes=[],
            wires=[]
        )
        
        # Input from sensor data
        input_node = self._create_node(
            1, "udl-input",
            {
                "name": f"{indicator_name} Data Input",
                "dataType": model_type,
                "refreshInterval": 60
            }
        )
        flow.nodes.append(input_node)
        
        # ML model node
        model_node = self._create_node(
            2, "ml-inference",
            {
                "name": f"{indicator_name} Model",
                "modelPath": f"/models/ccdm/{indicator_name}",
                "inputShape": [1, -1],
                "outputShape": [1, 2]
            }
        )
        flow.nodes.append(model_node)
        
        # Threshold node
        threshold_node = self._create_node(
            3, "switch",
            {
                "name": "Apply Thresholds",
                "property": "payload.score",
                "rules": [
                    {"t": "lt", "v": "0.3", "vt": "num"},
                    {"t": "btwn", "v": "0.3", "v2": "0.7", "vt": "num"},
                    {"t": "gte", "v": "0.7", "vt": "num"}
                ],
                "outputs": 3
            }
        )
        flow.nodes.append(threshold_node)
        
        # Status assignment nodes
        fail_node = self._create_node(
            4, "change",
            {
                "name": "Set FAIL",
                "rules": [{"t": "set", "p": "payload.status", "to": "FAIL"}]
            }
        )
        flow.nodes.append(fail_node)
        
        warning_node = self._create_node(
            5, "change",
            {
                "name": "Set WARNING",
                "rules": [{"t": "set", "p": "payload.status", "to": "WARNING"}]
            }
        )
        flow.nodes.append(warning_node)
        
        pass_node = self._create_node(
            6, "change",
            {
                "name": "Set PASS",
                "rules": [{"t": "set", "p": "payload.status", "to": "PASS"}]
            }
        )
        flow.nodes.append(pass_node)
        
        # Output node
        output_node = self._create_node(
            7, "kafka-producer",
            {
                "name": "Publish Indicator",
                "topic": f"welders.ss4.ccdm.{indicator_name}"
            }
        )
        flow.nodes.append(output_node)
        
        # Wire nodes together
        flow.wires = [
            [input_node["id"], model_node["id"]],
            [model_node["id"], threshold_node["id"]],
            [threshold_node["id"], fail_node["id"]],    # Output 1
            [threshold_node["id"], warning_node["id"]],  # Output 2
            [threshold_node["id"], pass_node["id"]],     # Output 3
            [fail_node["id"], output_node["id"]],
            [warning_node["id"], output_node["id"]],
            [pass_node["id"], output_node["id"]]
        ]
        
        # Deploy flow
        async with self.session.post(
            f"{self.config.base_url}/flow",
            json=flow.dict()
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
        return data["id"]
        
    async def get_flow_status(self, flow_id: str) -> Dict[str, Any]:
        """Get status of a Node-RED flow"""
        async with self.session.get(
            f"{self.config.base_url}/flow/{flow_id}"
        ) as response:
            response.raise_for_status()
            return await response.json()
            
    async def update_flow(self, flow_id: str, updates: Dict[str, Any]):
        """Update an existing flow"""
        async with self.session.put(
            f"{self.config.base_url}/flow/{flow_id}",
            json=updates
        ) as response:
            response.raise_for_status()
            
    async def inject_test_data(self, flow_id: str, data: Dict[str, Any]):
        """Inject test data into a flow"""
        inject_node = {
            "id": "test-inject",
            "type": "inject",
            "payload": json.dumps(data),
            "payloadType": "json",
            "once": True
        }
        
        # Add inject node to flow temporarily
        flow = await self.get_flow_status(flow_id)
        flow["nodes"].append(inject_node)
        
        await self.update_flow(flow_id, flow)
        
    async def monitor_flow_metrics(self, flow_id: str) -> Dict[str, Any]:
        """Get flow execution metrics"""
        async with self.session.get(
            f"{self.config.base_url}/flow/{flow_id}/metrics"
        ) as response:
            if response.status == 200:
                return await response.json()
            return {}


class CCDMWorkflowManager:
    """Manages all CCDM indicator workflows"""
    
    def __init__(self, node_red_service: NodeREDService):
        self.node_red = node_red_service
        self.indicators = [
            # Stability Indicators
            ("object_stability", "lstm"),
            ("stability_changes", "random_forest"),
            # Maneuver Indicators
            ("maneuvers_detected", "bilstm"),
            ("pattern_of_life", "temporal_mining"),
            # RF Indicators
            ("rf_detection", "cnn"),
            ("subsatellite_deployment", "multi_target"),
            # Compliance Indicators
            ("itu_fcc_compliance", "rule_based"),
            ("analyst_consensus", "ensemble"),
            # Signature Indicators
            ("optical_signature", "dnn"),
            ("radar_signature", "cnn3d"),
            # Stimulation
            ("system_response", "reinforcement"),
            # Physical Indicators
            ("area_mass_ratio", "physics_ml"),
            ("proximity_operations", "gnn"),
            # Tracking Indicators
            ("tracking_anomalies", "isolation_forest"),
            ("imaging_maneuvers", "behavioral"),
            # Launch Indicators
            ("launch_site", "geospatial"),
            ("un_registry", "nlp"),
            # Deception Indicators
            ("camouflage_detection", "multispectral"),
            ("intent_assessment", "game_theory")
        ]
        
    async def deploy_all_workflows(self):
        """Deploy all CCDM indicator workflows"""
        # Deploy main aggregation workflow
        main_flow_id = await self.node_red.deploy_ccdm_workflow()
        logger.info(f"Deployed main CCDM workflow: {main_flow_id}")
        
        # Deploy individual indicator workflows
        for indicator_name, model_type in self.indicators:
            try:
                flow_id = await self.node_red.create_indicator_workflow(
                    indicator_name,
                    model_type
                )
                logger.info(f"Deployed {indicator_name} workflow: {flow_id}")
            except Exception as e:
                logger.error(f"Failed to deploy {indicator_name} workflow: {e}")
                
    async def update_indicator_threshold(self, indicator_name: str, thresholds: Dict[str, float]):
        """Update thresholds for a specific indicator"""
        flow_id = f"{self.node_red.config.flow_prefix}-{indicator_name}"
        flow = await self.node_red.get_flow_status(flow_id)
        
        # Find and update threshold node
        for node in flow["nodes"]:
            if node["type"] == "switch" and node["name"] == "Apply Thresholds":
                node["rules"][0]["v"] = str(thresholds.get("fail", 0.3))
                node["rules"][1]["v"] = str(thresholds.get("warning_low", 0.3))
                node["rules"][1]["v2"] = str(thresholds.get("warning_high", 0.7))
                node["rules"][2]["v"] = str(thresholds.get("pass", 0.7))
                break
                
        await self.node_red.update_flow(flow_id, flow) 