bind = "0.0.0.0:8000"
workers = 4
threads = 2
worker_class = "sync"
timeout = 120  # Increased timeout for long-running operations
max_requests = 1000
max_requests_jitter = 50
keepalive = 5
