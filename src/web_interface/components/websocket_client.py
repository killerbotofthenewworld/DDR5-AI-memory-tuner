"""
WebSocket server for real-time updates
"""
import asyncio
import json
import logging
from typing import Dict, List, Any
import websockets
import threading
import time
from datetime import datetime


class DDR5WebSocketServer:
    """WebSocket server for real-time DDR5 monitoring"""
    
    def __init__(self, host: str = "localhost", port: int = 8502):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.server = None
        self.running = False
        self.data_cache = {}
        
    async def register_client(self, websocket, path):
        """Register a new WebSocket client"""
        client_id = f"client_{len(self.clients)}"
        self.clients[client_id] = websocket
        
        try:
            await websocket.send(json.dumps({
                "type": "connection",
                "client_id": client_id,
                "message": "Connected to DDR5 WebSocket server"
            }))
            
            # Send initial data
            if self.data_cache:
                await websocket.send(json.dumps({
                    "type": "initial_data",
                    "data": self.data_cache
                }))
            
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if not self.clients:
            return
            
        message = json.dumps({
            "type": "update",
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Store in cache
        self.data_cache.update(data)
        
        # Send to all clients
        disconnected_clients = []
        for client_id, websocket in self.clients.items():
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            del self.clients[client_id]
    
    def start_server(self):
        """Start the WebSocket server"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.server = websockets.serve(
                self.register_client,
                self.host,
                self.port
            )
            
            self.running = True
            loop.run_until_complete(self.server)
            loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        return thread
    
    def stop_server(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()


class RealTimeMonitor:
    """Real-time monitoring system for DDR5 parameters"""
    
    def __init__(self, websocket_server: DDR5WebSocketServer):
        self.ws_server = websocket_server
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Simulate DDR5 monitoring data
                data = self._collect_ddr5_data()
                
                # Send via WebSocket
                if self.ws_server.running:
                    asyncio.run(self.ws_server.broadcast_update(data))
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(5)  # Wait before retry
    
    def _collect_ddr5_data(self) -> Dict[str, Any]:
        """Collect real-time DDR5 data"""
        import random
        
        # Simulate real-time data collection
        # In production, this would interface with actual hardware
        return {
            "performance": {
                "bandwidth_gbps": round(random.uniform(50, 80), 2),
                "latency_ns": round(random.uniform(60, 90), 2),
                "throughput_mbps": round(random.uniform(45000, 75000), 0)
            },
            "thermal": {
                "temperature_c": round(random.uniform(35, 65), 1),
                "thermal_throttling": random.choice([True, False])
            },
            "power": {
                "vddq_v": round(random.uniform(1.05, 1.35), 3),
                "vpp_v": round(random.uniform(1.75, 2.05), 3),
                "power_consumption_w": round(random.uniform(8, 20), 1)
            },
            "stability": {
                "error_rate": round(random.uniform(0, 0.001), 6),
                "stability_score": round(random.uniform(85, 100), 1)
            },
            "timings": {
                "cl": random.randint(28, 40),
                "trcd": random.randint(28, 40), 
                "trp": random.randint(28, 40),
                "tras": random.randint(60, 80)
            }
        }


# WebSocket client-side JavaScript
WEBSOCKET_CLIENT_JS = """
<script>
class DDR5WebSocketClient {
    constructor(url = 'ws://localhost:8502') {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.callbacks = {};
    }
    
    connect() {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = (event) => {
                console.log('Connected to DDR5 WebSocket server');
                this.reconnectAttempts = 0;
                this.trigger('connected', event);
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.trigger('message', data);
                
                // Handle specific message types
                if (data.type === 'update') {
                    this.trigger('update', data.data);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('Disconnected from WebSocket server');
                this.trigger('disconnected', event);
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.trigger('error', error);
            };
            
        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.attemptReconnect();
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }
    
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }
    
    trigger(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => callback(data));
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Initialize WebSocket client
const ddr5WS = new DDR5WebSocketClient();

// Connect to server
ddr5WS.connect();

// Handle real-time updates
ddr5WS.on('update', (data) => {
    // Update UI elements with real-time data
    updatePerformanceMetrics(data.performance);
    updateThermalData(data.thermal);
    updatePowerData(data.power);
    updateStabilityData(data.stability);
    updateTimingData(data.timings);
});

function updatePerformanceMetrics(performance) {
    // Update performance charts and metrics
    if (performance.bandwidth_gbps) {
        updateElement('bandwidth-value', performance.bandwidth_gbps + ' GB/s');
    }
    if (performance.latency_ns) {
        updateElement('latency-value', performance.latency_ns + ' ns');
    }
}

function updateThermalData(thermal) {
    // Update thermal information
    if (thermal.temperature_c) {
        updateElement('temperature-value', thermal.temperature_c + '°C');
    }
}

function updatePowerData(power) {
    // Update power metrics
    if (power.vddq_v) {
        updateElement('vddq-value', power.vddq_v + ' V');
    }
    if (power.vpp_v) {
        updateElement('vpp-value', power.vpp_v + ' V');
    }
}

function updateStabilityData(stability) {
    // Update stability information
    if (stability.stability_score) {
        updateElement('stability-score', stability.stability_score + '%');
    }
}

function updateTimingData(timings) {
    // Update timing information
    Object.keys(timings).forEach(key => {
        updateElement(key + '-value', timings[key]);
    });
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        element.classList.add('pulse');
        setTimeout(() => element.classList.remove('pulse'), 1000);
    }
}
</script>
"""


def get_websocket_client_js():
    """Get the WebSocket client JavaScript code"""
    return WEBSOCKET_CLIENT_JS
