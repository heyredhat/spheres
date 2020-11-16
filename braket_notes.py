import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit

aws_account_id = boto3.client("sts").get_caller_identity()["Account"]

#device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
device = AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-8")
s3_folder = (f"amazon-braket-4462aa97a5a2", "Output")

bell = Circuit().h(0).cnot(0, 1)
task = device.run(bell, s3_folder, shots=100)
#print(task.result().measurement_counts)

from pytket import Circuit
from pytket.backends.braket import BraketBackend

circ = Circuit(2,2)
circ.H(0)
circ.CX(0,1)
#circ.Measure(0,0)
#circ.Measure(1,1)

backend = BraketBackend(\
	s3_bucket="amazon-braket-4462aa97a5a2",\
	s3_folder="Output",\
	device_type="quantum-simulator",\
	provider="amazon")
backend.compile_circuit(circ)

#print(backend.get_probabilities(circ, n_shots=500))
handle = backend.process_circuits([circ, circ], n_shots=500)
