#!/usr/bin/env python3
"""Simple HTTP server that listens on port 8081 for POST requests."""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json


def take_action(data: dict) -> dict:
    """
    Process the incoming request data.
    
    Args:
        data: The JSON payload from the POST request
        
    Returns:
        A response dictionary
    """
    # TODO: Implement your action logic here
    print(f"Received data: {data}")
    return {"status": "success", "message": "Action taken", "received": data}


class RequestHandler(BaseHTTPRequestHandler):
    """Handle incoming HTTP requests."""
    
    def do_POST(self):
        """Handle POST requests by calling take_action."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return
        
        # Call the take_action function
        result = take_action(data)
        
        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def run_server(port: int = 8081):
    """Start the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Server running on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
