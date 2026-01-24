#!/usr/bin/env python3
"""
Simple web server for QuanQonscious GRVQ-TTGCR Framework
Provides a basic web interface to interact with the quantum simulation framework.
"""

import http.server
import socketserver
import json
import os
import sys
from typing import Any, Dict, List

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maya_cipher import MayaCipher

class QuanQonsciousHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_index_page().encode())
        elif self.path == '/api/info':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            info = {
                "name": "QuanQonscious",
                "version": "0.1.0",
                "description": "GRVQ-TTGCR hybrid quantum-classical framework with Vedic sutra integration",
                "status": "running",
                "available_modules": [
                    "ansatz", "maya_cipher", "zpe_solver", "core_engine",
                    "deformulisation_engine", "primarysutra", "sulba"
                ]
            }
            self.wfile.write(json.dumps(info, indent=2).encode())
        elif self.path == '/api/modules':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            modules = self.get_available_modules()
            self.wfile.write(json.dumps(modules, indent=2).encode())
        elif self.path == '/api/hsqcp/metadata':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.get_hsqcp_metadata(), indent=2).encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/execute':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode())
                result = self.execute_command(data)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": str(e), "success": False}
                self.wfile.write(json.dumps(error_response).encode())
        elif self.path == '/api/hsqcp/run':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode())
                result = self.execute_hsqcp(data)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": str(e), "success": False}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_index_page(self):
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuanQonscious - GRVQ-TTGCR Framework</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card h3 {
            margin-top: 0;
            color: #fff;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            margin-right: 8px;
        }
        .module-list {
            list-style: none;
            padding: 0;
        }
        .module-list li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .module-list li:last-child {
            border-bottom: none;
        }
        .execute-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .section-title {
            margin-top: 0;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 14px;
        }
        .form-group input::placeholder, .form-group textarea::placeholder {
            color: rgba(255, 255, 255, 0.7);
        }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #45a049;
        }
        .output {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            padding: 15px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            opacity: 0.8;
            font-size: 0.9em;
        }
        .btn.secondary {
            background: #2196F3;
        }
        .btn.secondary:hover {
            background: #1e88e5;
        }
        .pill {
            display: inline-block;
            padding: 4px 10px;
            margin: 4px 6px 0 0;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.2);
            font-size: 0.85em;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
        }
        .result-card {
            background: rgba(0, 0, 0, 0.25);
            padding: 12px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>QuanQonscious</h1>
            <p>GRVQ-TTGCR Hybrid Quantum-Classical Simulation Framework</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3><span class="status-indicator"></span>System Status</h3>
                <p>Framework: <strong>Active</strong></p>
                <p>Version: <strong>0.1.0</strong></p>
                <p>Python Backend: <strong>Running</strong></p>
            </div>
            
            <div class="card">
                <h3>Systems & Domains</h3>
                <div id="system-tags"></div>
            </div>
            
            <div class="card">
                <h3>Industry Alignment</h3>
                <div id="industry-tags"></div>
            </div>
        </div>
        
        <div class="execute-section">
            <h3 class="section-title">One-Click Sutra Engine</h3>
            <p>Launch the full 29-sutra hybrid run (serial, concurrent, parallel) in one click and see how it maps to the platform systems and industries.</p>
            <div class="form-group">
                <label for="hsqcp-value">Input Seed Value</label>
                <input type="number" id="hsqcp-value" value="1.618" step="0.001">
            </div>
            <div class="form-group">
                <label for="hsqcp-mode">Execution Mode</label>
                <select id="hsqcp-mode">
                    <option value="hybrid">Hybrid</option>
                    <option value="classical">Classical</option>
                    <option value="quantum">Quantum</option>
                    <option value="maya_illusion">Maya Illusion</option>
                    <option value="sulba">Sulba</option>
                </select>
            </div>
            <div class="form-group">
                <label for="hsqcp-include">Filter Sutras (optional substring)</label>
                <input type="text" id="hsqcp-include" placeholder="e.g. maya, zpe, sulba">
            </div>
            <div class="form-group">
                <label for="hsqcp-precision">Precision</label>
                <input type="number" id="hsqcp-precision" value="64">
            </div>
            <div class="form-group">
                <label for="hsqcp-iterations">Max Iterations</label>
                <input type="number" id="hsqcp-iterations" value="128">
            </div>
            <button class="btn secondary" onclick="runHybridBundle()">Run Hybrid Sutra Bundle</button>
            <div id="hsqcp-output" class="output" style="display: none;"></div>
        </div>

        <div class="execute-section">
            <h3 class="section-title">Maya Cipher Operations</h3>
            <div class="form-group">
                <label for="cipher-key">Cipher Key (integer)</label>
                <input type="number" id="cipher-key" value="123456">
            </div>
            <div class="form-group">
                <label for="cipher-message">Message (UTF-8)</label>
                <textarea id="cipher-message" placeholder="Enter message"></textarea>
            </div>
            <button class="btn" onclick="encryptMessage()">Encrypt Message</button>
            <button class="btn secondary" onclick="decryptMessage()">Decrypt Ciphertext</button>
            <div id="cipher-output" class="output" style="display: none;"></div>
        </div>
        
        <div class="footer">
            <p>© 2025 QuanQonscious - Advanced Quantum-Classical Simulation Framework</p>
            <p>Author: Daniel James Elliot Meyer | Email: danmeyer85@gmail.com</p>
        </div>
    </div>

    <script>
        function populateMetadata() {
            fetch('/api/hsqcp/metadata')
                .then(response => response.json())
                .then(data => {
                    const systemTags = document.getElementById('system-tags');
                    const industryTags = document.getElementById('industry-tags');
                    systemTags.innerHTML = '';
                    industryTags.innerHTML = '';
                    data.systems.forEach(item => {
                        const span = document.createElement('span');
                        span.className = 'pill';
                        span.textContent = item;
                        systemTags.appendChild(span);
                    });
                    data.industries.forEach(item => {
                        const span = document.createElement('span');
                        span.className = 'pill';
                        span.textContent = item;
                        industryTags.appendChild(span);
                    });
                });
        }

        function runHybridBundle() {
            const outputDiv = document.getElementById('hsqcp-output');
            const payload = {
                value: parseFloat(document.getElementById('hsqcp-value').value),
                mode: document.getElementById('hsqcp-mode').value,
                include: document.getElementById('hsqcp-include').value,
                precision: parseInt(document.getElementById('hsqcp-precision').value, 10),
                max_iterations: parseInt(document.getElementById('hsqcp-iterations').value, 10)
            };
            outputDiv.style.display = 'block';
            outputDiv.textContent = 'Running full hybrid bundle...';
            fetch('/api/hsqcp/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                outputDiv.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                outputDiv.textContent = 'Error: ' + error.message;
            });
        }

        function encryptMessage() {
            const outputDiv = document.getElementById('cipher-output');
            const payload = {
                command: 'encrypt',
                key: document.getElementById('cipher-key').value,
                message: document.getElementById('cipher-message').value
            };
            outputDiv.style.display = 'block';
            outputDiv.textContent = 'Encrypting...';
            fetch('/api/execute', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                outputDiv.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                outputDiv.textContent = 'Error: ' + error.message;
            });
        }

        function decryptMessage() {
            const outputDiv = document.getElementById('cipher-output');
            const payload = {
                command: 'decrypt',
                key: document.getElementById('cipher-key').value,
                ciphertext: document.getElementById('cipher-message').value
            };
            outputDiv.style.display = 'block';
            outputDiv.textContent = 'Decrypting...';
            fetch('/api/execute', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                outputDiv.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                outputDiv.textContent = 'Error: ' + error.message;
            });
        }

        populateMetadata();
    </script>
</body>
</html>
        """
    
    def get_available_modules(self):
        modules = []
        python_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'web_server.py']
        for file in python_files:
            modules.append({
                "name": file[:-3],  # Remove .py extension
                "file": file,
                "status": "available"
            })
        return modules
    
    def execute_command(self, data):
        command = data.get('command', '')
        
        if command == 'info':
            return {
                "success": True,
                "data": {
                    "name": "QuanQonscious",
                    "version": "0.1.0",
                    "description": "GRVQ-TTGCR hybrid quantum-classical framework",
                    "python_version": sys.version,
                    "working_directory": os.getcwd(),
                    "available_files": [f for f in os.listdir('.') if f.endswith(('.py', '.txt'))]
                }
            }
        
        elif command == 'encrypt':
            key = int(data.get('key', 0))
            message = data.get('message', '').encode('utf-8')
            cipher = MayaCipher(key=key, rounds=4, use_time=False)
            ciphertext = cipher.encrypt_message(message, t=0.0).hex()
            return {"success": True, "ciphertext_hex": ciphertext}
        
        elif command == 'decrypt':
            key = int(data.get('key', 0))
            ciphertext = bytes.fromhex(data.get('ciphertext', ''))
            cipher = MayaCipher(key=key, rounds=4, use_time=False)
            plaintext = cipher.decrypt_message(ciphertext, t=0.0).decode('utf-8')
            return {"success": True, "plaintext": plaintext}
        
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

    def execute_hsqcp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from hybrid_sutra_platform import run_hybrid_bundle
        from sutra_repository import SutraMode

        value = float(data.get('value', 1.618))
        mode_value = str(data.get('mode', 'hybrid')).upper()
        include = data.get('include') or None
        precision = int(data.get('precision', 64))
        max_iterations = int(data.get('max_iterations', 128))
        max_workers = data.get('max_workers')
        mode = SutraMode[mode_value]
        bundle = run_hybrid_bundle(
            value,
            mode=mode,
            precision=precision,
            max_iterations=max_iterations,
            include=include,
            max_workers=max_workers,
        )
        return {
            "success": True,
            "bundle": bundle.to_dict(),
            "summary": {
                "serial": {
                    "aggregate": bundle.serial.aggregate,
                    "wall_time": bundle.serial.wall_time,
                },
                "concurrent": {
                    "aggregate": bundle.concurrent.aggregate,
                    "wall_time": bundle.concurrent.wall_time,
                },
                "parallel": {
                    "aggregate": bundle.parallel.aggregate,
                    "wall_time": bundle.parallel.wall_time,
                },
            },
            "systems": self.get_hsqcp_metadata()["systems"],
            "industries": self.get_hsqcp_metadata()["industries"],
        }

    def get_hsqcp_metadata(self) -> Dict[str, List[str]]:
        return {
            "systems": [
                "Sutra Execution Engine (29 sutras)",
                "Hybrid Quantum-Classical Simulator",
                "GRVQ / TGCR / ZPE Pipeline",
                "Maya Cipher Cryptography",
                "Performance & Timing Analytics",
            ],
            "industries": [
                "Quantum Computing & Simulation",
                "Defense & Secure Communications",
                "Financial Optimization & Trading",
                "Advanced Materials & Physics",
                "Healthcare Signal Modeling",
                "Energy Grid Optimization",
            ],
        }

def main():
    PORT = 3000
    Handler = QuanQonsciousHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"QuanQonscious Web Server running on port {PORT}")
            print(f"Access the application at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
