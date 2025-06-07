import asyncio
from log_manager import log

async def restart_program():
    cmd_stop = ["/bin/bash","/notebooks/stop_process.sh"]
    
    proc_stop = await asyncio.create_subprocess_exec(
            *cmd_stop,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    
    try:
        # read lines as they come in
        assert proc_stop.stdout is not None
        async for raw_line in proc_stop.stdout:
            line = raw_line.decode("utf-8").strip("\n")
            log.debug(line, flush=True)

    except asyncio.CancelledError:
        # if someone cancels the task, kill the subprocess
        proc_stop.kill()
        await proc_stop.wait()
        
    cmd_start = ["/bin/bash","/notebooks/start_process.sh"]
     
    proc_start = await asyncio.create_subprocess_exec(
            *cmd_start,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    
    try:
        # read lines as they come in
        assert proc_start.stdout is not None
        async for raw_line in proc_start.stdout:
            line = raw_line.decode("utf-8").strip("\n")
            log.debug(line, flush=True)

    except asyncio.CancelledError:
        # if someone cancels the task, kill the subprocess
        proc_start.kill()
        await proc_start.wait()