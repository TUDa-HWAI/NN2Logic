from fxpmath import Fxp
from quanto import qint8, qtype, quantize, Calibration, QTensor, freeze, QLinear
from torch import nn
import torch
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchmetrics.classification import MulticlassAccuracy
from collections import OrderedDict
import numpy as np
from torch.nn.functional import relu
from torch.nn.utils import fuse_linear_bn_eval
import json

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from nn2logic import InputEncoder, QTreeBuilder, QScales, FixedPoint, QLayer

torch.manual_seed(64)



def testModel(model, loader, device=torch.device('cpu')):
    metric = MulticlassAccuracy(2).to(device)
    
    model.eval()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        if isinstance(output, QTensor):
            output = output.dequantize()
        metric.update(output, target)

    print(metric.compute())


def modelCompute(model, loader, device=torch.device('cpu')):
    model.eval()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)


def trainModel(model, optimizer, lossF, loader, device=torch.device('cpu')):
    model.train()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, QTensor):
            output = output.dequantize()
        loss = lossF(output, target)
        loss.backward()
        optimizer.step()


class EncodingBuilder(object):
    def encNumerical(self, column, key):
        if not key in self.enc:
            self.enc[key] = MinMaxScaler(clip=True)
            return self.enc[key].fit_transform(column)

        return self.enc[key].transform(column)

    def __init__(self):
        self.enc = {}

        self.varNames = []
        self.C = InputEncoder()


    def apply(self, df):
        # transform columns independently
        interm = OrderedDict()
        createModel = len(self.varNames) == 0

        idx = 0
        roomVars = []
        for col in df.columns:
            if col in ['kitchen', 'livingroom', 'bedroom', 'office', 'kitchen_livingroom']:
                if createModel:
                    self.varNames.append(f"x_{idx}")
                    self.C.registerBinary(f"x_{idx}", 255)
                    roomVars.append(f"x_{idx}")
                    idx += 1

                interm[col] = df[[col]].to_numpy().reshape(-1)
                continue


            assert is_numeric_dtype(df[col])
            interm[col] = self.encNumerical(df[[col]], col).reshape(-1)
            
            if createModel:
                self.varNames.append(f"x_{idx}")
                self.C.registerInt(f"x_{idx}", 255)
                idx += 1

        if createModel:
            self.C.markBinariesOneHot(roomVars, True)
            self.C.update()

        # create DF using new columns
        return pd.DataFrame(interm)


encoder = EncodingBuilder()


class NidDataset(Dataset):
    def __init__(self, file, sep=','):
        data = pd.read_csv(file, sep=sep)
        data.drop(['datetime', 'room_number', 'apartment_number', 'room_type'], inplace=True, axis=1)  # irrelevant columns

        # separate into X and y
        X = data.drop(['occupancy_ground_truth'], axis=1)
        y = data[['occupancy_ground_truth']]

        # encode
        X = encoder.apply(X)

        # create tensor
        self.X = torch.tensor(X.values).float()
        self.y = torch.tensor(y.values).long().view(-1)  # use long if cuda cries

    def __len__(self):
        return self.X.size(dim=0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




class DumbLinear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, X):
        X = X.cpu().numpy()
        res = [(self.weight @ x + self.bias) for x in X]

        return torch.tensor(np.array(res))

    def export(self):
        return self.weight, self.bias


class DumbScaler(nn.Module):
    def __init__(self, scale, trackMax=False):
        super().__init__()
        self.scale = Fxp(scale, signed=False, n_frac=24)
        self.trackMax = trackMax
        self.max = 0

    def export(self):
        return [FixedPoint(s, self.scale.n_frac) for s in self.scale.val]

    def expQScales(self):
        return QScales(self.export(), 127)

    def forward(self, X):
        val = X * self.scale.astype(float)

        if self.trackMax:
            self.max = max(torch.max(torch.abs(val)).item(), self.max)

        return torch.trunc(val)




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(22, 10)
        self.lin2 = nn.Linear(10, 25)
        self.lin3 = nn.Linear(25, 2)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(25)

    def forward(self, x):
        x = self.lin1(x)
        x = relu(x)

        x = self.lin2(x)
        x = relu(x)

        x = self.lin3(x)
        return x


def layerBaseParams(state_dict, key):
    return {
        'weight' : model.state_dict()[f'{key}.weight._data'].cpu().to(torch.int64).numpy(),
        'weight_scale' : model.state_dict()[f'{key}.weight._scale'].view(-1).cpu().numpy(),
        'bias' : model.state_dict()[f'{key}.bias'].cpu().numpy()
    }



if __name__ == '__main__':
    # select device
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # dataset
    full = NidDataset('../datasets/occupancy/Dataset.csv', sep=',')
    train_set, test_set = random_split(full, [0.8, 0.2], torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)


    # create model
    state_dict = torch.load("occupancy.ckpt", map_location=torch.device(device))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    model = Model()
    model.eval()
    model.load_state_dict(state_dict)
    model.lin1 = fuse_linear_bn_eval(model.lin1, model.bn1)
    model.lin2 = fuse_linear_bn_eval(model.lin2, model.bn2)


    model.to(device)
    quantize(model, weights=qint8, activations=qint8)

    # calibrate model
    with Calibration():
        for idx, (X, y) in enumerate(train_loader):
            model(X.to(device))

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    lossF = nn.CrossEntropyLoss()
    for _ in range(8):
        trainModel(model, optimizer, lossF, train_loader, device)

    # testing the model
    testModel(model, test_loader, device)


    # quantize
    model.eval()
    freeze(model)
    state_dict = model.state_dict()


    # obtain path from retrained model
    retr = []
    retr.append((model.lin1.weight.dequantize().detach().cpu().numpy(),
        model.lin1.bias.detach().cpu().numpy()))
    retr.append((model.lin2.weight.dequantize().detach().cpu().numpy(),
        model.lin2.bias.detach().cpu().numpy()))
    retr.append((model.lin3.weight.dequantize().detach().cpu().numpy(),
        model.lin3.bias.detach().cpu().numpy()))


    print("create quantized hybrid code")
    amm_max = 127.0

    input_scale = 1.0 / 255

    # own calibration routine
    layers = []
    ## initial scaling of the dataset
    mod = DumbScaler(1 / input_scale, trackMax=True)
    modelCompute(mod, train_loader)
    print("initial scaling", mod.max)
    layers.append(mod)


    ## first layer
    p1 = layerBaseParams(state_dict, 'lin1')
    lin1 = DumbLinear(
        p1['weight'], 
        np.round(p1['bias'] / (input_scale * p1['weight_scale']))
    )
    layers.append(lin1)
    layers.append(nn.ReLU())

    ## first requant
    r1i = DumbScaler(input_scale * p1['weight_scale'], trackMax=True)
    layers.append(r1i)

    modelCompute(nn.Sequential(*layers), train_loader)  # determine maximum

    ### substitute scaler with new magic number, if necessary
    if True:#r1i.max > amm_max:
        r1 = DumbScaler((amm_max * input_scale * p1['weight_scale']) / r1i.max)
        layers[-1] = r1

        ## adapt input_scale
        input_scale = (amm_max * input_scale) / r1i.max
        print("input_scale I", input_scale)


    ## second layer
    p2 = layerBaseParams(state_dict, 'lin2')
    lin2 = DumbLinear(
        p2['weight'],
        np.round(p2['bias'] / (input_scale * p2['weight_scale']))
    )

    layers.append(lin2)
    layers.append(nn.ReLU())

    ## second requant
    r2i = DumbScaler(input_scale * p2['weight_scale'], trackMax=True)
    layers.append(r2i)

    modelCompute(nn.Sequential(*layers), train_loader)  # determine maximum
    
    ### substitute scaler with new magic number, if necessary
    if True:#r2i.max > amm_max:
        r2 = DumbScaler((amm_max * input_scale * p2['weight_scale']) / r2i.max)
        layers[-1] = r2

        ## adapt input_scale
        input_scale = (amm_max * input_scale) / r2i.max

        print("input_scale II", input_scale)


    ## third layer
    p3 = layerBaseParams(state_dict, 'lin3')
    lin3 = DumbLinear(
        p3['weight'],
        np.round(p3['bias'] / (input_scale * p3['weight_scale']))
    )

    layers.append(lin3)

    ## third requant
    r3i = DumbScaler(input_scale * p3['weight_scale'], trackMax=True)
    layers.append(r3i)

    modelCompute(nn.Sequential(*layers), train_loader)  # determine maximum
    
    ### substitute scaler with new magic number
    r3 = DumbScaler((127.0 * input_scale * p3['weight_scale']) / r3i.max)
    layers[-1] = r3


    model.to(torch.device('cpu'))

    dumb = nn.Sequential(*layers)
    testModel(dumb, test_loader)

    print("determine tree paths")
    # create tree
    ctrain = np.array([train_set.dataset[train_set.indices[idx]][0].tolist() for idx in range(len(train_set))])
    ctrain *= 255
    print(type(lin1.export()[0]), True, r1.expQScales())
    print(QLayer(lin1.export()[0], lin1.export()[1], True, r1.expQScales()))
    print("about to build layers")
    layers = [
        QLayer(*lin1.export(), True, r1.expQScales()),
        QLayer(*lin2.export(), True, r2.expQScales()),
        QLayer(*lin3.export(), False, r3.expQScales())
    ]
    print("got them layers")
    tree = QTreeBuilder(layers, encoder.C, ctrain.astype(int))
    print("got tree")
    j = tree.toJson()
    
    # save training set
    targets = []
    data = []

    for x, y in train_set:
        out = mod(x)
        data.append([int(a) for a in out.tolist()])
        targets.append(y.item())

    j["targets"] = targets
    j["data"] = data

    # save json
    with open('qat-w-samples.json', 'w') as f:
        json.dump(j, f)

    # save testing set
    with open("dataset.h", "w") as f:
        f.write("#include <stdint.h>\n#include <stdbool.h>\n\n")
        f.write("const size_t datasetSize = {};\n\n".format(len(test_set)))
        f.write("const uint8_t dataset[{}][{}] = {{\n".format(len(test_set), test_set[0][0].size(dim=0)))

        targets = []

        for x, y in test_set:
            out = mod(x)

            f.write("   { ")
            f.write(", ".join([str(int(a)) for a in out.tolist()]))
            f.write(" },\n")

            targets.append(y)

        f.write("};\n")

        # target
        f.write("const uint8_t target[] = { ")
        f.write(", ".join([str(t.item()) for t in targets]))
        f.write(" };\n")



    # tree.print("qat-tree.json")

    # h = HybridGen([
    #         (*lin1.export(), r1.export(), True),
    #         (*lin2.export(), r2.export(), True),
    #         (*lin3.export(), r3.export(), False),
    #     ], tree.getPaths())

    # h.render("/Users/daniel/Documents/research/asm/hybrid.c")


    # renderNetwork([
    #         (*lin1.export(), r1.export(), True),
    #         (*lin2.export(), r2.export(), True),
    #         (*lin3.export(), r3.export(), False),
    #     ], "/Users/daniel/Documents/research/asm/test.c")

    