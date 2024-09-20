from qiskit import QuantumCircuit
 
def ghz_circuit(n):
    """
    input: n number of qubits
    output: quantum circuit generating GHZ state of n qubits
    """
    # Create a circuit with a register of n qubits
    circ = QuantumCircuit(n)
    # H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
    circ.h(0)
    for i in range(1,n):
    # CX (CNOT) gate on control qubit 0 and target qubit i resulting in a GHZ state.
        circ.cx(0, i)
    
    # # Draw the circuit
    # circ.draw('mpl')
    return circ
