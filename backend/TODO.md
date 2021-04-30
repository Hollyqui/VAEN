
# TODO

## Abstraction Layer

- Create nodes for graph/tree
- Nodes contain pointers to inputs and outputs (thereby connecting the nodes)
- Nodes either contain the tensorflow data or have uids
- If they have uids, those uids will be the key in a dictionary containing the tensorflow data (thereby avoiding cache poisoning)
- Any other data for display purposes (Names, position and so on)
- They have an input/output flag (could be a bool; input: True - output: False - hidden: None)
- Need to figure out how to determine whether all inputs have arrived at a given node (array of bools? One for every input?)
- Connect nodes in different threads
- Figure out some clever way to store complete graph structure (for easy access to nodes) (array of arrays maybe? (each nester array is one layer))

## Interface

- Somehow interface with popular ML libraries (like pytorch or tensorflow)
