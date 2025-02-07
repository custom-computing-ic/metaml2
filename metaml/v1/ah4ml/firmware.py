from artisan import *
#from meta_cl import *
from heterograph.hgraph import HGraph
from .layer import Layer, Stream, Elem
from functools import lru_cache
from typing import List, Tuple
from colorama import Fore, Style

import os.path as osp
import glob
import clang.cindex as ci
import re
import subprocess

class Firmware:
    def __init__(self, rootdir):
        self.rootdir = rootdir
        firmware_srcpath = osp.join(rootdir, 'firmware')
        cppsrcs = glob.glob(osp.join(firmware_srcpath, '*.cpp'))
        if len(cppsrcs) == 0:
            raise RuntimeError(f"cannot find source file (.cpp) in '{firmware_srcpath}'!")
        if len(cppsrcs) != 1:
            raise RuntimeError(f"multiple source files found (.cpp) in '{firmware_srcpath}'!")

        self.__streams = {} # {stream name: Stream}
        self.__layers = {}  # {layer name: Layer}
        self.__arch = [ ]   # elements in order s0 => l0 => s1 => l1
        self.__arch_map = { } # 'name' => elem
        self.input = None
        self.output = None

        self.srcname = cppsrcs[0]
        params_src = osp.join(osp.dirname(self.srcname), 'parameters.h')
        defines_src = osp.join(osp.dirname(self.srcname), 'defines.h')
        ap_types_dir = osp.join('firmware', 'ap_types')
        defines_header = osp.join('firmware', 'defines.h')

        self.ast = Ast(f"{self.srcname} {params_src} {defines_src} -xc++ -I {ap_types_dir} -include {defines_header}", workdir=rootdir, preprocessor=True)
        self.ast.parse_pragmas()
        print(f"[i] firmware source: '{self.srcname}'")
        self.__build()

    def reset(self):
        for e in self.arch:
            self[e].reset()

    def __repr__(self):
        center = int(max([len(e) for e in self.arch]) / 2) - 1
        s = ""
        for elem_name in self.arch:
            elem = self[elem_name]
            space = ' ' * (center-int(len(elem_name)/2))
            ename = f"{Fore.LIGHTCYAN_EX}{space}{elem_name}" if elem.is_layer else f"{Fore.LIGHTYELLOW_EX}{space}{elem_name}"
            arrow = "" if self.input == elem_name else f"\n{' ' * center}⟱\n"
            s = s + f"{arrow}{ename}{Style.RESET_ALL}"
        return s

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__arch_map[item]
        elif isinstance(item, int):
            return self.__arch_map[self.__arch[item]]

    def __build(self):
        # find firmware function
        fns = self.ast.query("firmware_fn{FunctionDecl}")
        if len(fns) != 1:
            raise RuntimeError(f"expecting one firmware function in '{self.srcname}', found: {len(fns)}!")

        fn = fns[0].firmware_fn # 'firmware_fn' is based on the query
        print(f"[i] firmware function: '{fn.name}'")

        self.__get_iotype(fn)
        print(f"[i] IO type: {self.iotype}")

        if self.iotype == 'dataflow':
           self.__build_streams_dataflow(fn) # self.__streams
        elif self.iotype == 'pipeline':
           self.__build_streams_pipeline(fn) # self.__streams
        else:
            raise RuntimeError("[x] IO type '{self.iotype}' not implemented!")


        self.__build_layers(fn)  # self.__layers
        self.__build_arch()      # self.__arch

    def __add_element(self, is_layer, name, obj):
        # sanity check: ensure elements (streams and layer) have unique names
        if name in self.__streams or name in self.__layers:
            raise RuntimeError(f"[x] internal error: element {name} already defined!")
        if is_layer:
            self.__layers[name] = obj
        else:
            self.__streams[name] = obj
        self.__arch_map[name] = obj

    def __get_iotype(self, fn):
        pragmas = fn.module.pragmas
        for pragma in pragmas:
            # Each pragma has the format: ([token list], token_start, token_end)
            token_list = pragma[0]
            if len(token_list) > 1:
                if token_list[0] == 'HLS' and token_list[1] == 'DATAFLOW':
                    self.iotype = 'dataflow'
                    return
                if token_list[0] == 'HLS' and token_list[1] == 'PIPELINE':
                    self.iotype = 'pipeline'
                    return

        raise RuntimeError("[x] cannot infer firmware iotype!")

    def __build_streams_dataflow(self, fn):
        stream_io = fn.query("param{ParmDecl}={1}>ns{NamespaceRef}", fd=1,
                              where=lambda ns: ns.spelling == 'hls')

        for p in stream_io:
            name = p.param.name
            stream = Stream(p.param)
            self.__add_element(False, name, stream)

        stream_decls = fn.query("stmt{DeclStmt} ={1}> var{VarDecl} ={1}> ns{NamespaceRef}", fd=2,
                           where=lambda ns: ns.spelling == 'hls')
        for decl in stream_decls:
            name = decl.var.name
            stream = Stream(decl.var)
            self.__add_element(False, name, stream)


    def __build_streams_pipeline(self, fn):

        stream_io = fn.query("param{ParmDecl}={1}>typeref{TypeRef}", fd=1)

        for p in stream_io:
            name = p.param.name
            stream = Stream(p.param)
            self.__add_element(False, name, stream)

        stream_decls = fn.query("stmt{DeclStmt} ={1}> var{VarDecl} ={1}> typeref{TypeRef}", fd=2)

        for decl in stream_decls:
            name = decl.var.name
            stream = Stream(decl.var)
            self.__add_element(False, name, stream)

    def __build_layers(self, fn):

        # first get the names of layers which are in comments
        # this is associted with the function block
        block = fn.query("block{CompoundStmt}", fd=1)[0].block
        layer_linenum = {} # line number => layer name
        for token in block.tokens:
            if token.kind == ci.TokenKind.COMMENT:
                comment = token.spelling
                if comment[0:2] == "//":
                    comment = comment[2:].strip()
                    # ensure it is word (identifier)
                    if re.fullmatch(r"\w+", comment):
                        layer_linenum[token.location.line] = comment

        # 'fd' let's us choose the exact level position of calls inside
        # the function
        layers = fn.query("layer{CallExpr}", fd=2)
        for row in layers:
            # ensure we find the layer name through the line number
            line = row.layer.location.line
            if line not in layer_linenum:
                raise RuntimeError(f"cannot find name for this call: '{row.layer.name}' ({row.layer.location})!")

            layer_name = layer_linenum[line]
            layer = Layer(layer_name, row.layer)
            self.__add_element(True, layer_name, layer)

    def __build_arch(self):
        g = HGraph()
        # create dependency graph
        # element name => vertex index

        gmap = { }
        for l in self.__layers:
            layer = self.__layers[l]
            deps = []
            for elem in [layer.in_stream, l, layer.out_stream]:
                if elem not in gmap:
                    vx = g.add_vx()
                    g.pmap[vx]['name'] = elem
                    gmap[elem] = vx
                else:
                    vx = gmap[elem]
                deps.append(vx)
            g.add_edge(deps[0], deps[1])
            g.add_edge(deps[1], deps[2])


        self.__arch = [ g.pmap[vx]['name'] for vx in g.vertices ]

        # sanity checks
        if g.num_vx != len(self.__layers) + len(self.__streams):
            raise RuntimeError("[x] internal error: architecture not consistent with C++ code!")

        for vx in g.vertices:
            num_in_vx = g.num_in_vx(vx)
            if num_in_vx > 1:
                raise RuntimeError(f"[x] more than one input for node '{g.pmap[vx]['name']}': expecting 1 or 0!")
            elif num_in_vx == 0:
                if self.input is not None:
                   raise RuntimeError(f"[x] multiple sources found in graph: expecting 1!")
                else:
                    self.input = g.pmap[vx]['name']

            num_out_vx = g.num_out_vx(vx)
            if num_out_vx > 1:
                raise RuntimeError(f"[x] more than one output for node '{g.pmap[vx]['name']}': expecting 1 or 0!")
            elif num_out_vx == 0:
                if self.output is not None:
                   raise RuntimeError(f"[x] multiple sinks found in graph: expecting 1!")
                else:
                    self.output = g.pmap[vx]['name']

        if self.input is None:
            raise RuntimeError("[x] no source found!")

        if self.output is None:
            raise RuntimeError("[x] no sink found!")

    @property
    def n_virtual_layers(self) -> int:
        return sum([e.has_weights for e in self])

    @property
    def n_reuse_factors(self) -> int:
        return sum([e.exists('reuse_factor') for e in self])

    @property
    def reuse_factors(self) -> Tuple[int, ...]:
        output = []
        for e in self:
            if e.exists("reuse_factor"):
                output.append(int(e.get("reuse_factor")))

        return tuple(output)

    @reuse_factors.setter
    def reuse_factors(self, rfs: Tuple[int, ...]):
        assert len(rfs) == self.n_reuse_factors, "Number of reuse factors given must match number in model"

        rfs = iter(rfs)
        rf = None

        for e in self:
            if e.exists("reuse_factor"):
                rf = next(rfs)
                e.set("reuse_factor", rf)

    @property
    def reuse_factor_layer_names(self):
        output = []
        for e in self:
            if e.exists("reuse_factor"):
                output.append(e.config_name)

        return output


    def set_virtual_layer_precisions(self, configuration: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]) -> None:
        assert len(configuration) == self.n_virtual_layers, "Size of configuration must match number of virtual layers in model"

        # Create configuration iterator
        configuration = iter(configuration)
        weight, bias, result = None, None, None

        # Iterate through all model elements apart from input and output streams, ensuring consistent interfacing with model.
        # Any layers prior to first weights layer are also skipped.
        for elem_name in self.arch[:-1]:
            elem = self[elem_name]


            # If the element is a weights-layer, update the preicsion of its weights and biases
            if elem.has_weights:
                weight, bias, result = next(configuration)
                # self._set_if_exists(elem, "mult_config.weight_t", ap_fixed(weight))
                # self._set_if_exists(elem, "mult_config.bias_t", ap_fixed(bias))
                self._set_if_exists(elem, "weight_t", ap_fixed(weight))
                self._set_if_exists(elem, "bias_t", ap_fixed(bias))
                #print("elem_name: ", elem_name)
                #print(weight)
                continue

            # If weight has not been initialized, i.e. first weights layer not encountered, we continue
            if weight is None:
                continue

            # If the element is a stream, its precision is the result type of the previous layer
            if elem.is_stream:
                elem.set(re.sub("ap_fixed<[0-9]*,[ ]?[0-9]*>", ap_fixed(result), elem.get()))
                #print("elem_name: ", elem_name)
                #print(re.sub("ap_fixed<[0-9]*,[ ]?[0-9]*>", ap_fixed(result), elem.get()))
                #print(elem.get())
                #print("elem_name: ", elem_name)
                #print(result)
                continue

            # If the layer is a pooling layer, its precision is the result type of the previous layer
            if "pooling" in elem.cpp_class:
                #self._set_if_exists(elem, "accum_t", ap_fixed(result))
                pass

    @staticmethod
    def _set_if_exists(elem: Elem, attribute: str, value: str) -> None:
        if elem.exists(attribute):
            elem.set(attribute, value)

    @property
    def arch(self):
        return self.__arch

    @property
    @lru_cache()
    def layers(self):
        return [ e for e in self.__arch if e in self.__layers ]

    @property
    @lru_cache()
    def streams(self):
        return [ e for e in self.__arch if e in self.__streams ]

    def gen(self, dirname=None, *, build=False, inplace=None):
        self.ast.reset() # resets changes

        # add SYNTHESIS define
        module = self.ast.sources[0].module
        ret=module.query("d{DeclStmt}={1}>v{VarDecl}", where=lambda v:v.name == 'loaded_weights')
        if len(ret) != 1:
            raise RuntimeError("[x] could not remove non-synth code!")
        ret[0].d.instrument(action=Action.before, code="\n#ifndef __SYNTHESIS__\n")
        ret=module.query("c{IfStmt}")
        if len(ret) != 1:
            raise RuntimeError("[x] could not remove non-synth code!")
        ret[0].c.instrument(action=Action.after, code="\n#endif\n")

        # restore pragma, which is broken by align
        def fn_pragmas(pragma):
            if pragma[0:3] == ['HLS', 'INTERFACE', 'axis' ] and len(pragma) == 8:
                return f"#pragma HLS INTERFACE axis port={pragma[5]}\n#pragma HLS INTERFACE axis port={pragma[7]}"
            else:
                return True
        module.instrument(Action.pragmas, fn=fn_pragmas)


        if dirname is not None and inplace is not None:
            raise RuntimeError("[x] cannot use 'dirname' and 'inplace' parameters at the same time!")

        # perform instrumentation here
        for elem in self.arch:
            self[elem].apply_changes()

        if inplace is None or inplace == False:
            new_ast = self.ast.clone(name=dirname, changes=True, align=True, parse=False)
        else:
            self.ast.unmanaged()
            self.ast.sync(commit=False)
            new_ast = self.ast

        if build:
            subprocess.run(["vivado_hls",  "build_prj.tcl"], cwd=new_ast.workdir)

        return new_ast.workdir


def ap_fixed(precision: Tuple[int, int]) -> str:
    return f"ap_fixed<{precision[0]},{precision[1]}>"
