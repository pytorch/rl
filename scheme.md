

```
      Sender side                                       Receiver Side

  +-----------------------+
  | MCollector.__init__() |
  +-----------------------+
             |
  +----------------------+                         +------------------------+
  | init_on_sender (all) |                         | init_on_receiver (all) |
  +----------------------+                         +------------------------+
             |                                                  | 
             |                                                  | 
  +------------------------------+                 +------------------------------+
  | synchronize weights (policy) |<---  BLOCK  --->| synchronize weights (policy) |
  +------------------------------+                 +------------------------------+
             |                                                  | 
             |                                     +------------------------------+
             |                                     | SyncDataCollector.__init__() |
             |                                     +------------------------------+
             |                                                  |
  +--------------------------------------+         +--------------------------------------+
  | synchronize weights (all non-policy) |<-BLOCK->| synchronize weights (all non-policy) |
  +--------------------------------------+         +--------------------------------------+
  
  +---------------+                                +------------------+
  | scheme.send() | -----------------------------> | scheme.receive() | # Does not need collector
  +---------------+                                +------------------+
                                                            |
                                                   +------------------------+
                                                   | scheme.apply_weights() |
                                                   +------------------------+

```

Sync transfer with generic (no RPC)

```
       Main (DCollector)            Intermediate (MCollector)           Sub (SCollector)
  +------------------------+       +------------------------+         +------------------------+
  | update_policy_weight_()|       | update_policy_weight_()| ------> | scheme.receive()       |
  +------------------------+       +------------------------+         +------------------------+
             |
  +------------------------+       +----------------------+
  | scheme.send(weights)   |       |
  +------------------------+

```

In the sync DCollector case:

- Uses dist.send on DCollector side, dist.recv on MCollector side
- The scheme between MCollector and SCollector can be established via a kwarg in collector_kwargs
- We manually call MCollector.update_policy_weights_ with the new copy
- MCollector never sees the scheme from the DCollector