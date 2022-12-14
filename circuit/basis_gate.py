import torch
import numpy as np
from torch._C import dtype
import math

def kronecker(A, B):
    if not isinstance(A, torch.Tensor):
        return B
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

Z = torch.tensor([
    [1.0, 0],
    [0, -1.0]
], dtype=torch.cfloat)

X = torch.tensor([
    [0, 1.0],
    [1.0, 0]
], dtype=torch.cfloat)

Y = torch.tensor([
    [0, -1.0j],
    [1.0j, 0]
], dtype=torch.cfloat)

H = math.sqrt(2)/2*torch.tensor([
    [1, 1],
    [1, -1]
], dtype=torch.cfloat)

I = torch.eye(2, dtype=torch.cfloat)

def RX(phi):
    gate = torch.zeros((2, 2), dtype=torch.cfloat)
    gate[0, 0], gate[0, 1], gate[1, 0], gate[1, 1] = torch.cos(phi/2), torch.complex(torch.tensor(0, dtype=torch.float32), -torch.sin(phi/2)), torch.complex(torch.tensor(0, dtype=torch.float32), -torch.sin(phi/2)), torch.cos(phi/2)
    return gate.to(phi.device)

def RY(phi):
    gate = torch.zeros((2, 2), dtype=torch.cfloat)
    gate[0, 0], gate[0, 1], gate[1, 0], gate[1, 1] = torch.cos(phi/2), -torch.sin(phi/2), torch.sin(phi/2), torch.cos(phi/2)
    return gate.to(phi.device)

def RZ(phi):
    gate = torch.zeros((2, 2), dtype=torch.cfloat)
    gate[0, 0], gate[1, 1] = torch.complex(torch.cos(phi/2), -torch.sin(phi/2)), torch.complex(torch.cos(phi/2), torch.sin(phi/2))
    return gate.to(phi.device)

def Rot(alpha, beta, theta):
    return torch.mm(torch.mm(RX(alpha), RY(beta)), RZ(theta))


def State00():
    state = torch.zeros((2, 2), dtype=torch.cfloat)
    state[0, 0] = 1
    return state

def State11():
    state = torch.zeros((2, 2), dtype=torch.cfloat)
    state[1, 1] = 1
    return state

def CNOT(wires=[0, 1]):
    if wires[0] < wires[1]:
        first, second = State00(), State11()
        for i in range(wires[0], wires[1]):
            if i == wires[1] - 1:
                first = kronecker(first, torch.eye(2))
                second = kronecker(second, X)
            else:
                first = kronecker(first, torch.eye(2))
                second = kronecker(second, torch.eye(2))
        return first + second
    else:
        first, second = torch.eye(2), X
        for i in range(wires[1], wires[0]):
            if i == wires[0] - 1:
                first = kronecker(first, State00())
                second = kronecker(second, State11())
            else:
                first = kronecker(first, torch.eye(2))
                second = kronecker(second, torch.eye(2))
        return first + second

RGate = {
    'RX': RX,
    'RY': RY,
    'RZ': RZ
}

def CRR(phi, wires=[0, 1], name='RX'):
    if wires[0] < wires[1]:
        first, second = State00(), State11()
        for i in range(wires[0], wires[1]):
            if i == wires[1] - 1:
                first = kronecker(first, torch.eye(2))
                second = kronecker(second, RGate[name](phi))
            else:
                first = kronecker(first, torch.eye(2))
                second = kronecker(second, torch.eye(2))
        return first + second
    else:
        first, second = torch.eye(2), RGate[name](phi)
        for i in range(wires[1], wires[0]):
            if i == wires[0] - 1:
                first = kronecker(first, State00)
                second = kronecker(second, State11)
            else:
                first = kronecker(first, torch.eye(2))
                second = kronecker(second, torch.eye(2))
        return first + second

if __name__=='__main__':
    state = torch.zeros((2**2, 1), dtype=torch.cfloat)
    state[3] = 1
    H = math.sqrt(2)/2*torch.tensor([
        [1, 1],
        [1, -1]
    ], dtype=torch.cfloat)
    gate = CNOT(wires=[1,0])
    # gate = kronecker(H, H) @ gate @ kronecker(H, H)
    print(gate@state)