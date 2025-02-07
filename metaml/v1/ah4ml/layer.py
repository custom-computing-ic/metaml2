import clang.cindex as ci
from .spec import Spec
from colorama import Fore, Style
import os.path as osp

class Elem:
    def __init__(self, *, node, name):
        self.node = node
        self.name = name
        self.__spec = Spec(self)

    def apply_changes(self):
        self.__spec.apply()

    def __getitem__(self, item):
        return self.__spec[item]
        
    @property
    def is_stream(self):
        return False

    @property
    def is_layer(self):
        return False

    @property
    def has_weights(self):
        return False

    def __repr__(self):
        s = f"{Fore.LIGHTYELLOW_EX}[{self.name}]{Style.RESET_ALL}\n"
        for attr in self.__spec.attributes:
            entry = self.__spec.attributes[attr]
            attr_type = "[T]" if entry[0] == 0 else "[C]"
            val = f"{Fore.LIGHTMAGENTA_EX}{entry[3]}" if entry[4] is None else f"{Fore.LIGHTRED_EX}{entry[4]}*"
            attr_name = "" if attr == "" else f" {attr}"
            s += f"   {Fore.LIGHTBLACK_EX}+---{Fore.LIGHTBLUE_EX}{attr_name}{Fore.LIGHTBLACK_EX} {attr_type}{Style.RESET_ALL}: {val}{Style.RESET_ALL}\n"
        return s

    def get(self, name=None):
        if name is None:
            name = ""
        return self.__spec.get(name)

    def set(self, arg0=None, arg1=None):
        if arg0 is None and arg1 is None:
            raise RuntimeError("[x] bad use of 'set'")
        if arg1 is None:
            return self.__spec.set("", arg0)     
        else:   
            return self.__spec.set(arg0, arg1)

    def reset(self):
        self.__spec.reset()         

    def _configure_spec(self, datatype):
        self.__spec.configure(datatype)

    def exists(self, name):
        return name in self.__spec.attributes
    
class Stream(Elem):
    def __init__ (self, node):
        super().__init__(node=node, name=node.name)      
        self.__build()

    @property
    def is_stream(self):
        return True        

    def __build(self):

        tp = None
        _type = None
        if self.node.isentity("VarDecl"):
            if len(self.node.children) == 2 and self.node.child(0).isentity('TypeRef'):
                # pipeline
                 _type = self.node.child(0).type
            else:
                # dataflow
                tp = self.node.type
        elif self.node.isentity("ParmDecl"):

            if len(self.node.children) == 1 and self.node.child(0).isentity('TypeRef'):
                # pipeline
                 _type = self.node.child(0).type
            else:
                #dataflow
                tp = self.node.type.get_pointee()  

        if tp is not None:            
            num_args = tp.get_num_template_arguments()
            if num_args != 1:
                raise RuntimeError(f"[x] Stream '{self.name}' ({self.node.location}) has the wrong number of template of arguments. Expecting '1', got '{num_args}'!")

            _type = tp.get_template_argument_type(0)

        if _type is None:
                raise RuntimeError(f"[x] Stream '{self.name}' ({self.node.location}) has a consistency error: cannot determine type!")

        self._configure_spec(_type)

            
class Layer(Elem):
    def __init__(self, name, node):   
        if not node.isentity("CallExpr"):
            raise RuntimeError(f"[x] Layer not implemented for AST entity: '{node.entity}'!")

        super().__init__(node=node, name=name)

        self.cpp_class = None      # C++ class name 
        self.in_stream = None      # name
        self.in_type = None        # name 
        self.out_stream = None     # name
        self.out_type = None       # name
        self.config_name = None    # name
        self.config_class = None   # C++ class name
        self.weights = None        # name
        self.biases = None         # name
        self.__build()

    @property
    def is_layer(self):
        return True

    @property
    def has_weights(self):
        #return self.cpp_class in ("dense", "conv_2d_cl", "normalize") # batch normalization
        return self.cpp_class in ("dense", "conv_2d_cl")

    # we need the streams to build each layer
    def __build(self):
        # accepted pattern: 3 DeclRefExpr or 5 DeclRefExpr (with weights and biases)
        children = self.node.children
        if len(children) not in [3, 5]:
            raise RuntimeError('[x] unexpected layer pattern (wrong number of nodes)!')        
        if not all([child.isentity('DeclRefExpr') for child in children]):
            raise RuntimeError('[x] unexpected layer pattern (wrong type of nodes)!')

        self.cpp_class = children[0].name
        self.in_stream = children[1].name
        self.out_stream = children[2].name
        if len(children) == 5:
            self.weights = children[3].name
            self.biases = children[4].name


        # check template types: needs 3 types
        reftypes = [ n for n in self.node.child(0).children if n.isentity("TypeRef")]
        
        if len(reftypes) != 3:
            raise RuntimeError(f"[x] internal error - do not support layer '{self.cpp_class}'")

        if not all([n.type.kind == ci.TypeKind.TYPEDEF for n in [reftypes[0], reftypes[1]]]):
            raise RuntimeError(f"[x] internal error - do not support layer '{self.cpp_class}': first two template type arguments must be typedef!")
            
        if reftypes[2].type.kind != ci.TypeKind.RECORD:
            raise RuntimeError(f"[x] internal error - do not support layer '{self.cpp_class}': last template type argument must be a record!")
        
        self.in_type = reftypes[0].spelling
        self.out_type = reftypes[1].spelling
        self.config_name = reftypes[2].spelling.split(' ')[-1]
        
        # find config name
        decl_config = reftypes[2].type.get_declaration()
        loc = decl_config.location
        srcname = osp.join('firmware', osp.basename(loc.file.name))
        cdecl = self.node.ast.find(source=srcname, line=loc.line, column=loc.column)
        base_cls = cdecl.query("b{CxxBaseSpecifier}={1}>t{TypeRef}", fd=1)
        if len(base_cls) != 1:
            raise RuntimeError("[x] cannot find config name information!")
        _base_type = base_cls[0].t

        self.config_class = _base_type.unparse().strip()
        self._configure_spec(reftypes[2].type)


        

