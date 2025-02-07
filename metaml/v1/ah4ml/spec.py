import clang.cindex as ci
import os.path as osp
from collections import OrderedDict
from artisan import *

class Spec:
    def __init__(self, elem):
        self.elem = elem
        self.ast = elem.node.ast
        self.datatype = None

        # construct1, construct2... allows to reconstruct root-cnode, value can be changed by the user
        # 'name' => [0=typedef, 1=constant, root-cnode, None or [construct1, construct2, ...], 'original value', None or 'modified value']
        self.attributes = OrderedDict()

    def reset(self):
        for attrib in self.attributes:
            self.attributes[attrib][4] = None

    def get(self, name):
        modified_val = self.attributes[name][4]
        return self.attributes[name][3] if modified_val is None else modified_val


    def set(self, name, value):
        self.attributes[name][4] = str(value)


    def configure(self, datatype):
        if self.datatype:
            raise RuntimeError("[x] internal error: spec already configured!")

        self.datatype = datatype

        if self.datatype.kind not in [ci.TypeKind.TYPEDEF, ci.TypeKind.RECORD]:
           raise RuntimeError(f"[x] Cannot create spec - type '{self.datatype.kind}' not supported!")

        decl = self.datatype.get_declaration()

        loc = decl.location
        srcname = osp.join('firmware', osp.basename(loc.file.name))
        cnode = self.ast.find(source=srcname, line=loc.line, column=loc.column)

        # sanity check
        typemap = {ci.TypeKind.TYPEDEF: "TypedefDecl", ci.TypeKind.RECORD: "StructDecl" }
        if typemap[self.datatype.kind] != cnode.entity:
            raise RuntimeError(f"[x] internal error: expecting '{typemap[self.datatype.kind]}' got '{cnode.entity}'!")

        self.__build(cnode)

    def __build(self, cnode, attr_name=None, prefix=None):
        if prefix is None:
            prefix = ""
        if cnode.isentity("TypedefDecl"):
            # expand typedef
            base_type = cnode.type
            if base_type.kind != ci.TypeKind.TYPEDEF:
                raise RuntimeError("[x] internal error processing typedef!")
            while base_type.kind == ci.TypeKind.TYPEDEF:
               type_decl = ci.conf.lib.clang_getTypeDeclaration(base_type)
               loc = type_decl.location
               srcname = osp.join('firmware', osp.basename(loc.file.name))
               # update cnode to point to the right typedef
               cnode = self.ast.find(source=srcname, line=loc.line, column=loc.column)
               base_type = ci.conf.lib.clang_getTypedefDeclUnderlyingType(type_decl)

            if base_type.kind != ci.TypeKind.RECORD:
                # normal case: typedef refers to basic type
                if attr_name is None:
                    attr_name = ""
                name = ("" if prefix == "" else f"{prefix}.") + attr_name

                self.__add_entry(attrib_type=0, name=name, cnode=cnode, constructs=None, value=base_type.spelling)
            else:
                sdecl = self.ast.query("sdecl{StructDecl}", where=lambda sdecl: sdecl.name == base_type.spelling)
                if len(sdecl) != 1:
                    raise RuntimeError("f[x] could not find struct declaration: {base_type.spelling}!")
                prefix = ("" if prefix == "" else f"{prefix}.") + cnode.name
                cnode = sdecl[0].sdecl
                self.__build(cnode, prefix=prefix)

        elif cnode.isentity("StructDecl"):
            # constants
            vdecl = cnode.query("decl{VarDecl}", fd=1)
            for v in vdecl:
                code = v.decl.unparse().split("=")
                if len(code) == 2 and code[0].startswith("static const"):
                    name = ("" if prefix == "" else f"{prefix}.") + v.decl.name
                    self.__add_entry(attrib_type=1, name=name, cnode=v.decl, constructs=code[0].strip(), value=code[1].strip())

            # typedef
            tdef = cnode.query("tdef{TypedefDecl}", fd=1)
            for t in tdef:
                cnode = t.tdef
                self.__build(cnode, attr_name=cnode.name, prefix=prefix)


    def __add_entry(self, *, name, attrib_type, cnode, constructs, value):
        # sanity check
        if name in self.attributes:
            raise RuntimeError(f"[x] internal error: attempting to add an attribute name which already exists: '{name}'!")

        self.attributes[name] = [attrib_type, cnode, constructs, value, None]


    def apply(self):
        for attrib in self.attributes:
            entry = self.attributes[attrib]

            if entry[4] is not None:
                elem_type = "stream" if self.elem.is_stream else "layer"
                attrib_desc = f"{attrib} <= " if attrib != "" else ""
                print(f"[i] instrumenting {elem_type} '{self.elem.name}': {attrib_desc}{entry[4]}")

                cnode = entry[1]
                if entry[0] == 0: # typedef
                    cnode.instrument(action=Action.replace, code=f"typedef {entry[4]} {cnode.name}")
                elif entry[0] == 1: # constant
                    cnode.instrument(action=Action.replace, code=f"{entry[2]} = {entry[4]}")














