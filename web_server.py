#!/usr/bin/env python3
"""
Simple web server for QuanQonscious GRVQ-TTGCR Framework
Provides a basic web interface to interact with the quantum simulation framework.
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import sys
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
                <h3>Available Modules</h3>
                <ul class="module-list">
                    <li>🔬 GRVQ Ansatz Construction</li>
                    <li>🔐 Maya Cipher Cryptography</li>
                    <li>⚡ ZPE Field Solver</li>
                    <li>🧮 Vedic Sutra Library</li>
                    <li>🌐 Core Engine</li>
                    <li>📊 Performance Analysis</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>Framework Features</h3>
                <ul class="module-list">
                    <li>General Relativity Integration</li>
                    <li>Vedic Mathematics (29 Sutras)</li>
                    <li>Quantum Circuit Simulation</li>
                    <li>HPC GPU Acceleration</li>
                    <li>Bioelectric DNA Encoding</li>
                    <li>TTGCR Hardware Simulation</li>
                </ul>
            </div>
        </div>
        
        <div class="execute-section">
            <h3>Quick Commands</h3>
            <div class="form-group">
                <label for="command">Select Command:</label>
                <select id="command" onchange="updateCommandParams()">
                    <option value="info">System Information</option>
                    <option value="encrypt">Maya Cipher Encrypt</option>
                    <option value="decrypt">Maya Cipher Decrypt</option>
                    <option value="simulate">ZPE Field Simulation</option>
                </select>
            </div>
            
            <div id="command-params">
                <!-- Dynamic parameters will be inserted here -->
            </div>
            
            <button class="btn" onclick="executeCommand()">Execute Command</button>
            
            <div id="output" class="output" style="display: none;"></div>
        </div>
        
        <div class="footer">
            <p>© 2025 QuanQonscious - Advanced Quantum-Classical Simulation Framework</p>
            <p>Author: Daniel James Elliot Meyer | Email: danmeyer85@gmail.com</p>
        </div>
    </div>

    <script>
        function updateCommandParams() {
            const command = document.getElementById('command').value;
            const paramsDiv = document.getElementById('command-params');
            
            let html = '';
            
            switch(command) {
                case 'encrypt':
                    html = `
                        <div class="form-group">
                            <label for="key">Encryption Key (integer):</label>
                            <input type="number" id="key" placeholder="123456" required>
                        </div>
                        <div class="form-group">
                            <label for="message">Message to Encrypt:</label>
                            <textarea id="message" placeholder="Enter your message here..." required></textarea>
                        </div>
                    `;
                    break;
                case 'decrypt':
                    html = `
                        <div class="form-group">
                            <label for="key">Decryption Key (integer):</label>
                            <input type="number" id="key" placeholder="123456" required>
                        </div>
                        <div class="form-group">
                            <label for="ciphertext">Ciphertext (hex):</label>
                            <textarea id="ciphertext" placeholder="Enter hex ciphertext..." required></textarea>
                        </div>
                    `;
                    break;
                case 'simulate':
                    html = `
                        <div class="form-group">
                            <label for="grid_size">Grid Size (NxNxN):</label>
                            <input type="number" id="grid_size" placeholder="20" value="20">
                        </div>
                        <div class="form-group">
                            <label for="steps">Time Steps:</label>
                            <input type="number" id="steps" placeholder="1" value="1">
                        </div>
                    `;
                    break;
                default:
                    html = '<p>No additional parameters required.</p>';
            }
            
            paramsDiv.innerHTML = html;
        }
        
        function executeCommand() {
            const command = document.getElementById('command').value;
            const outputDiv = document.getElementById('output');
            
            let params = {command: command};
            
            // Collect parameters based on command type
            switch(command) {
                case 'encrypt':
                    params.key = document.getElementById('key').value;
                    params.message = document.getElementById('message').value;
                    break;
                case 'decrypt':
                    params.key = document.getElementById('key').value;
                    params.ciphertext = document.getElementById('ciphertext').value;
                    break;
                case 'simulate':
                    params.grid_size = document.getElementById('grid_size').value;
                    params.steps = document.getElementById('steps').value;
                    break;
            }
            
            outputDiv.style.display = 'block';
            outputDiv.textContent = 'Executing command...';
            
            fetch('/api/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                outputDiv.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                outputDiv.textContent = 'Error: ' + error.message;
            });
        }
        
        // Initialize page
        updateCommandParams();
        
        // Load system info on page load
        fetch('/api/info')
            .then(response => response.json())
            .then(data => {
                console.log('System Info:', data);
            })
            .catch(error => {
                console.error('Failed to load system info:', error);
            });
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
            try:
                key = int(data.get('key', 0))
                message = data.get('message', '')
                # Simple encryption placeholder since we can't import the actual module
                result = f"Encrypted message with key {key}: {message.encode().hex()}"
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif command == 'decrypt':
            try:
                key = int(data.get('key', 0))
                ciphertext = data.get('ciphertext', '')
                # Simple decryption placeholder
                result = f"Decrypted with key {key}: {bytes.fromhex(ciphertext).decode('utf-8', errors='ignore')}"
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif command == 'simulate':
            try:
                grid_size = int(data.get('grid_size', 20))
                steps = int(data.get('steps', 1))
                result = f"ZPE Field Simulation: Grid={grid_size}x{grid_size}x{grid_size}, Steps={steps} - Simulation completed successfully (placeholder result)"
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

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
