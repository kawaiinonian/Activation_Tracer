import torch
from torch import nn
from typing import Union, List, Optional
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Activation_tracer:
    def __init__(self, model: nn.Module, trace_layers: Union[nn.Module, List[nn.Module], str, List[str]], 
                default_save_path: str='./') -> None:
        '''
            model: The model will be traced.
            trace_layers: Which layer you want trace, can be instance or class and list of instance or
            list of class. If instance, tracer just trace the instance(s) passed in. If class, tracer
            trace all the layers if they are instance of the class(es) passed in.
            default_save_path: The default path for figure saving, can be changed when generate a new 
            figure.
        '''
        self.model = model
        self.handlers = []
        self.default_save_path = default_save_path
        self.activation_pool = {}
        self.layer_name_dict = {}
        all_layers = list(model.named_modules())
        for (name, layer) in all_layers:
            self.layer_name_dict[layer] = name

        # the hook function
        def get_activation(model: nn.Module, input:torch.tensor, output:torch.tensor):
            self.activation_pool[model][1].append(output.clone().detach().cpu().numpy())

        if not isinstance(trace_layers, list):
            trace_layers = [trace_layers]

        if isinstance(trace_layers[0], nn.Module):
            for layer in trace_layers:
                handle = layer.register_forward_hook(get_activation)
                self.handlers.append(handle)
                self.activation_pool[layer] = [self.layer_name_dict[layer],[]]

        else:
            for (name, layer) in all_layers:
                if isinstance(layer, tuple(trace_layers)):
                    handle = layer.register_forward_hook(get_activation)
                    self.handlers.append(handle)
                    self.activation_pool[layer] = [name, []]

    def clear_pool(self):
        for k, v in self.activation_pool.items():
            v[1].clear()

    def export_pool(self):
        return deepcopy(self.activation_pool)

    def get_figure(self, mode:Optional[str]=None, input: Optional[List[dict]]=None, 
                   save_path: Optional[str]=None, with_hash: Optional[bool]=False):
        '''
        mode: Should be 'seperate', 'all_layer' or 'constractive'. If seperate, generate 
        figure for every layer in bar form seperately, elif all_layer, generate a heat map 
        for all layer passed in, elif constractive, generate a figure to constract a layer's 
        output under different input, and default build a all_layer figure for total model.
        input: A list for layer input.
        save_path: Default option is the path passed in when initialize.
        with_hash: Add a layer's hash in figure's name, not recommanded if not neccesary.
        '''

        # generate figure for every layer in bar form
        def seperate(input: List[dict], save_path: str):
            input = input[0]
            for layer, (name, values) in input.items():
                path = save_path + 'seperate_' + name
                if with_hash:
                    path += str(layer.__hash__())
                path += '.png'

                averange = np.zeros(values[0].shape, dtype=float)
                for value in values:
                    averange += value
                averange /= len(values)
                averange = averange.mean(axis=0)
                
                if len(averange.shape) == 1:
                    get_bar_plot(averange, path)
                # if layer has a special dimension, like conv, build a heat map
                elif len(averange.shape) == 2:
                    get_heat_map(averange, path)
                else:
                    raise NotImplementedError("Only 1D and 2D are supported now")

        # generate a figure to constract a layer's output under different input, to be done
        def constractive():
            pass
        # generate a heat map for all layer passed in
        def all_layer(input: List[dict], save_path: str):
            input = input[0]
            path = save_path + 'all_layers'
            path += '.png'
            datas = []
            pad_length = 0
            for layer, (_, values) in input.items():
                averange = np.zeros(values[0].shape, dtype=float)
                for value in values:
                    averange += value
                averange /= len(values)
                averange = averange.mean(axis=0)
                if len(averange.shape) == 2:
                    averange = averange.flatten()
                datas.append(averange)
                if len(averange) > pad_length:
                    pad_length = len(averange)
            for i, data in enumerate(datas):
                datas[i] = np.pad(data, (0, pad_length - len(data)), 'constant', constant_values=0)
            datas = np.array(datas)
            get_heat_map(datas, path)

        # if no layer passed in, generate an all-layer heat map for total model, to be done
        def default():
            pass
        def get_bar_plot(data: Union[np.ndarray, List[np.ndarray]], path: str):
            if isinstance(data, list):
                raise NotImplementedError()
            f, ax = plt.subplots(figsize = (6, 15))
            fig_data = {'neural': [str(i) for i in range(len(data))], 'activation': data, }
            sns.barplot(data = fig_data, x = 'activation', y = 'neural', color='b')
            # ax.legend(ncol=2, loc="lower right", frameon=True)
            f.savefig(path)
        def get_heat_map(data: Union[np.ndarray, List[np.ndarray]], path: str):
            if isinstance(data, list):
                raise NotImplementedError()
            f, ax = plt.subplots(figsize = (15, 15))
            fig_data = pd.DataFrame(data, [str(i) for i in range(len(data))])
            sns.heatmap(fig_data, ax=ax)
            f.savefig(path)
        
        
        func_dict = {
            "seperate": seperate,
            "constractive": constractive,
            "all_layer": all_layer,
            "default": default,
        }
        if not save_path:
            save_path = self.default_save_path
        if not mode:
            func_dict["default"](save_path)
        else:
            func_dict[mode](input, save_path)