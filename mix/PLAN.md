#Plan to Mix distributed-llm with air-llm

1. align distributed-llama to python
2. clone a airllm
3. make every model distributed to all nodes. (Shared-Storage Zero-Data Movement)
4. mix airllm + distributed-llama
5. under this target, re-organize / rewrite bottleneck to c/c++

# Distributed-AirLLM Project Tracking

## Current Phase: [1. Align distributed-llama to python]
- [ ] Sub-task 1.1: Create a Distributed-Llama worker in Python.
- [ ] Sub-task 1.2: Implement Layer-wise offset calculation.
- [ ] Sub-task 1.3: [TODO] Define the binary protocol for Activation exchange.

## Roadmap
1. **Python Alignment** [In Progress]
2. **AirLLM Integration** [Pending]
3. **Zero-Data Movement Logic** [Pending]
4. **Bottleneck C++ Rewrite** [Pending]

