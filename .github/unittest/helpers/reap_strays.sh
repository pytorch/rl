#!/usr/bin/env bash
# On OSDC the step runs under run_with_env_secrets.py, which drains our stdout
# until EOF. EOF only arrives once every process holding the write end has
# closed it, so a single xdist worker, Ray actor or Xvfb that outlives the tests
# keeps the job alive until its 120-minute timeout, long after the script exits.
# Under linux_job_v2 the surrounding `docker run` reaped these for us.

echo "::group::Processes still alive before exit"
ps -eo pid,ppid,etimes,rss,args --sort=-rss | head -40 || true
echo "::endgroup::"

# Anything still running that is not this shell or ps itself.
strays="$(pgrep -f 'pytest|ray::|Xvfb' 2>/dev/null | grep -v "^$$\$" || true)"
if [ -n "${strays}" ]; then
  echo "Reaping strays holding the step open: ${strays}"
  # shellcheck disable=SC2086
  kill -TERM ${strays} 2>/dev/null || true
  sleep 5
  # shellcheck disable=SC2086
  kill -KILL ${strays} 2>/dev/null || true
fi
