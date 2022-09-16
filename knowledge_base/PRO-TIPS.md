# Pro-tips

## Training on a cluster
- RL is known to be CPU-intensive in some instances. Even when running a few
    environments in parallel, you can see a great speed-up by asking for more cores on your cluster
    than the number of environments you're working with (twice as much for instance). This
    is also and especially true for environments that are rendered (even if they are rendered on GPU). 
